import os
import sys
import time
import numpy as np

import torch
import torch.distributed as dist

import deep_ep
import utils
from utils import init_dist, bench, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

import alltoall

try:
    from paperf import profile_torch
    has_paperf = True
except ImportError:
    has_paperf = False

profile = True
profile = profile and has_paperf


def init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input=False):
    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)

    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    if dump_input:
        utils.dump(x, 'x', local_rank)
        utils.dump(x_pure_rand, 'x_pure_rand', local_rank)
        utils.dump(x_e4m3, 'x_e4m3', local_rank)

        utils.dump(topk_idx, 'topk_idx', local_rank)
        utils.dump(topk_weights, 'topk_weights', local_rank)
        utils.dump(topk_weights_pure_rand, 'topk_weights_pure_rand', local_rank)

    return x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand


def load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts):
    def _load_tensor(rank, name, idx, typehint="tensor"):
        dump_dir = "/root/paddlejob/workspace/env_run/liuyiqun/outputs/ds_8nodes"
        filename = f"{dump_dir}/{idx}_{name}_rank{rank}.npy"
        if typehint == "tensor":
            x_np = np.load(filename)
            if x_np.dtype == np.uint16:
                x = torch.tensor(x_np, device='cuda').view(torch.bfloat16)
            elif x_np.dtype in [np.float32, np.int32, np.int64, np.bool, np.int8]:
                x = torch.tensor(x_np, device='cuda')
            else:
                assert False, f'{name}: {x_np.dtype}'
            return x
        else:
            assert False, f'invalid typehint: {typehint}'

    input_tensors = []
    for i in range(20):
        x = _load_tensor(rank, "dispatch_x", i + 1)
        topk_idx = _load_tensor(rank, "topk_idx", i + 1)
        topk_weights = _load_tensor(rank, "topk_weights", i + 1)
        input_tensors.append({"x": x, "topk_idx": topk_idx, "topk_weights": topk_weights})

    return input_tensors


def test_main(num_sms: int, local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: dist.ProcessGroup, use_random_input, dump_input):
    # Settings
    num_tokens = 4096
    hidden = 7168
    num_topk_groups = min(num_nodes, 4)
    num_topk = 8
    num_experts = (256 // num_ranks) * num_ranks

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    if use_random_input:
        if profile:
            profile_torch.push_record_event(f"init_random_tensors")

        x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand = init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input)

        if profile:
            profile_torch.pop_record_event()
    else:
        if profile:
            profile_torch.push_record_event(f"load_dumped_tensors")

        input_tensors = load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts)

        x = input_tensors[0]["x"]
        topk_idx = input_tensors[0]["topk_idx"]
        topk_weights = input_tensors[0]["topk_weights"]

        if profile:
            profile_torch.pop_record_event()

    if buffer is None:
        #buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=False)
        buffer = alltoall.get_buffer(group, hidden * 2)

    if profile:
        profile_torch.push_record_event(f"get_dispatch_layout")

    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    t = bench(group, lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print()

    if profile:
        profile_torch.pop_record_event()

    if profile:
        profile_torch.push_record_event(f"barrier")

    torch.distributed.barrier()
    #group.barrier()

    if profile:
        profile_torch.pop_record_event()

    time.sleep(1)

    # Config
    # rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    rdma_buffer_size = 128
    nvl_buffer_size = 288

    current_x = x
    handle = None

    nvl_chunk_size = 20
    rdma_chunk_size = 28

    config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
    if profile:
        profile_torch.push_record_event(f"Dispatch_Config({config_str})")

    #dispatch_config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
    dispatch_config = deep_ep.Buffer.get_dispatch_config(group.size())

    dispatch_num_nvl_bytes = dispatch_config.get_nvl_buffer_size_hint(hidden * 2, group.size())
    dispatch_num_rdma_bytes = dispatch_config.get_rdma_buffer_size_hint(hidden * 2, group.size())

    if handle is not None:
        dispatch_args = {'x': current_x, 'handle': handle, 'config': config}
    else:
        dispatch_args = {
            'x': current_x,
            'num_tokens_per_rank': num_tokens_per_rank,
            'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
            'is_token_in_rank': is_token_in_rank,
            'num_tokens_per_expert': num_tokens_per_expert,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'config': dispatch_config
        }
    result_times = bench(group, lambda: buffer.dispatch(**dispatch_args))
    gpu_time = result_times[0]
    cpu_time = result_times[3]

    if profile:
        profile_torch.pop_record_event()

    if local_rank == 0:
        print(f'[rank={rank}] Dispatch: SMs {num_sms}, nvl_chunk_size {nvl_chunk_size}, nvl_buffer_size {nvl_buffer_size}, rdma_chunk_size {rdma_chunk_size}, rdma_buffer_size {rdma_buffer_size}, num_nvl_bytes {dispatch_num_nvl_bytes}, num_rdma_bytes {dispatch_num_rdma_bytes}; gpu_time: {gpu_time:.5f} s, cpu_time: {cpu_time:.5f} s')

    #group.barrier()

    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    nvl_chunk_size = 1
    rdma_chunk_size = 20
    config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
    if profile:
        profile_torch.push_record_event(f"Combine_Config({config_str})")

    #combine_config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
    combine_config = deep_ep.Buffer.get_combine_config(group.size())

    combine_num_nvl_bytes = combine_config.get_nvl_buffer_size_hint(hidden * 2, group.size())
    combine_num_rdma_bytes = combine_config.get_rdma_buffer_size_hint(hidden * 2, group.size())

    combine_args = {
        'x': recv_x,
        'handle': handle,
        'config': combine_config
    }
    result_times = bench(group, lambda: buffer.combine(**combine_args))
    gpu_time = result_times[0]
    cpu_time = result_times[3]

    if profile:
        profile_torch.pop_record_event()

    if local_rank == 0:
        print(f'[rank={rank}] Combine: SMs {num_sms}, nvl_chunk_size {nvl_chunk_size}, nvl_buffer_size {nvl_buffer_size}, rdma_chunk_size {rdma_chunk_size}, rdma_buffer_size {rdma_buffer_size}, num_nvl_bytes {combine_num_nvl_bytes}, num_rdma_bytes {combine_num_rdma_bytes}; gpu_time: {gpu_time:.5f} s, cpu_time: {cpu_time:.5f} s')


def test_loop(local_rank: int, num_local_ranks: int):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    if profile:
        profile_torch.switch_profile(0, 0, 1)

    #buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=False)

    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    use_random_input = False
    dump_input = False
    for i in (20, ):
        test_main(i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, None, group, use_random_input, dump_input)

    if profile:
        profile_torch.switch_profile(1, 0, 1)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
