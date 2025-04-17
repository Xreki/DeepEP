import os
import sys
import time
import numpy as np
import paddle
import paddle.distributed as dist
from typing import Optional


def init_dist(num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    print(f"num_nodes: {num_nodes}, node_rank: {node_rank}")
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_parallel_env() 
    #dist.init_process_group(
    #    backend='nccl',
    #    init_method=f'tcp://{ip}:{port}',
    #    world_size=num_nodes * num_local_ranks,
    #    rank=node_rank * num_local_ranks + local_rank
    #)

    rank = dist.get_rank()
    local_rank = rank % num_local_ranks

    paddle.set_default_dtype(paddle.bfloat16)
    #paddle.set_default_device('cuda')
    #paddle.set_device(f"cuda:{local_rank}")

    return local_rank, rank, dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


#def calc_diff(x: torch.Tensor, y: torch.Tensor):
#    x, y = x.double() + 1, y.double() + 1
#    denominator = (x * x + y * y).sum()
#    sim = 2 * (x * y).sum() / denominator
#    return (1 - sim).item()


def per_token_cast_to_fp8(x: paddle.Tensor):
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    # m, n = x.shape
    m = x.shape[0]
    n = x.shape[1]
    x_view = x.view([m, -1, 128])
    x_amax = x_view.abs().cast(paddle.float32).amax(axis=2).view([m, -1]).clip(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).cast(paddle.float8_e4m3fn).view([m, n]), (x_amax / 448.0).view([m, -1])


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.cast(paddle.float32).view([x_fp8.shape[0], -1, 128])
    x_scales = x_scales.view([x_fp8.shape[0], -1, 1])
    return (x_fp32 * x_scales).view(x_fp8.shape).cast(paddle.bfloat16)


def inplace_unique(x: paddle.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = paddle.zeros([x.shape[0], num_slots + 1], dtype=x.dtype).to(x.place)
    # bin_count.scatter_add_(1, x_padded, paddle.ones_like(x_padded))
    bin_count.put_along_axis_(axis=1, indices=x_padded, values=paddle.ones_like(x_padded), reduce='add', include_self=True)

    bin_count = bin_count[:, :num_slots]
    sorted_bin_count = paddle.sort(bin_count, axis=-1, descending=True)
    sorted_bin_idx = paddle.argsort(bin_count, axis=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = paddle.sort(sorted_bin_idx, descending=True, axis=-1)
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.shape[1])
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: paddle.Tensor, group_idx: paddle.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view([num_tokens, num_groups, -1])
    mask = paddle.zeros([num_tokens, num_groups], dtype=paddle.int64)
    #mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    mask = mask.put_along_axis_(axis=1, indices=group_idx, values=1).cast(paddle.float32)
    mask = mask.unsqueeze(-1).expand_as(scores)
    return (scores * mask).view([num_tokens, num_experts])


def bench(group, fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    paddle.device.synchronize()
    cache = paddle.empty([int(256e6 // 4)], dtype=paddle.int32)

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    paddle.distributed.barrier(group)
    paddle.device.synchronize()

    cpu_start = time.time()
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    paddle.device.synchronize()
    cpu_runtime = time.time() - cpu_start

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times), cpu_runtime / num_tests


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tupled = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, '')) / scale)
                        break
                break
    return tuple(kernel_times) if is_tupled else kernel_times[0]


dtype2str = {
    paddle.float32: "_orgi_fp32",
    paddle.int32: "_orgi_int",
    paddle.int64: "_orgi_int64",
    paddle.bool: "_orgi_bool",
    paddle.bfloat16: "_orgi_bf16",
    paddle.float8_e4m3fn: "_orgi_fp8"
}


def dump(x, name, local_rank):
    dump_dir = "/root/paddlejob/workspace/env_run/liuyiqun/outputs/paddle_dump"
    name = dump_dir + "/" + name
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.float32 or x.dtype == paddle.int32 or x.dtype == paddle.int64 or x.dtype == paddle.bool:
            y = x.numpy()
        elif x.dtype == paddle.bfloat16:
            y = x.view('uint16').numpy()
        elif x.dtype == paddle.float8_e4m3fn:
            y = x.view(paddle.uint8).numpy()
        else:
            assert False, f'{name}: {x.dtype} {x}'
            # name += dtype2str[x.dtype]
        np.save(f"{name}_rank{local_rank}.npy",y)
    elif isinstance(x, tuple):
        y, y_scale = x
        assert y.dtype == paddle.float8_e4m3fn
        assert y_scale.dtype == paddle.float32
        y_dump = y.view(paddle.uint8).numpy()
        y_scale_dump = y_scale.numpy()
        # np.save(f"{name}{dtype2str[y.dtype]}_value_rank{local_rank}.npy", y_dump)
        # np.save(f"{name}{dtype2str[y_scale.dtype]}_scale_rank{local_rank}.npy", y_scale_dump)
        np.save(f"{name}_value_rank{local_rank}.npy", y_dump)
        np.save(f"{name}_scale_rank{local_rank}.npy", y_scale_dump)
    elif isinstance(x, list):
        y = np.asarray(x)
        np.save(f"{name}_rank{local_rank}.npy", y)
    elif x is None:
        np.save(f"{name}_rank{local_rank}.npy", np.zeros(5))
    else:
        assert False, f'{name}: {x}'


def retrive_dtype(name):
    if "_orgi_fp32" in name:
        return paddle.float32
    elif "_orgi_int" in name:
        return paddle.int32
    elif "_orgi_int64" in name:
        return paddle.int64
    elif "_orgi_bool" in name:
        return paddle.bool
    elif "_orgi_bf16" in name:
        return paddle.bfloat16
    elif "_orgi_fp8" in name:
        return paddle.float8_e4m3fn
    else:
        assert False, f"{name} with wrong dtype"


def load(name, local_rank, typehint="tensor"):
    dump_dir = '/root/paddlejob/workspace/env_run/liuyiqun/outputs/torch_dump'
    name = dump_dir + "/" + name
    print(f"[local_rank={local_rank}] load {name}")
    # orig_dtype = retrive_dtype(name)
    # name += dtype2str[x.dtype]
    if typehint == "tensor":
        x_np = np.load(f'{name}_rank{local_rank}.npy')
        if x_np.dtype == np.uint16:
            x = paddle.to_tensor(x_np).view(paddle.bfloat16)
        elif x_np.dtype == np.uint8:
            x = paddle.to_tensor(x_np).view(paddle.float8_e4m3fn)
        else:
            x = paddle.to_tensor(x_np)
        return x
    elif typehint == "tuple":
        y_np = np.load(f'{name}_value_rank{local_rank}.npy')
        y_scale_np = np.load(f'{name}_scale_rank{local_rank}.npy')
        y = paddle.to_tensor(y_np).view(paddle.float8_e4m3fn)
        y_scale = paddle.to_tensor(y_scale_np)
        return (y, y_scale)
    else:
        assert False, f'invalid typehint: {typehint}'
