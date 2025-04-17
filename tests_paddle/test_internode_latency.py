import os
import sys
import time
import numpy as np

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.distributed.communication.group import Group
import random

import fused_a2a
import utils

try:
    from paperf import profile_paddle
    has_paperf = True
except ImportError:
    has_paperf = False


profile = False
profile = profile and has_paperf


def print_tensor_info(t, name):
    #print(f"-- {name}: data_ptr={t.untyped_storage().data_ptr()}, shape={t.size()}, dtype={t.dtype}")
    print(f"-- {name}: shape={t.size()}, dtype={t.dtype}")


def init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input=False):
    # Random data
    x = paddle.ones(shape=[num_tokens, hidden], dtype=paddle.bfloat16) * rank
    x_pure_rand = paddle.randn(shape=[num_tokens, hidden], dtype=paddle.bfloat16)
    x_e4m3 = utils.per_token_cast_to_fp8(x)

    scores = paddle.randn(shape=[num_tokens, num_experts], dtype=paddle.float32).abs() + 1
    group_scores = scores.view([num_tokens, num_nodes, -1]).amax(axis=-1)
    group_idx = paddle.topk(group_scores, num_topk_groups, axis=-1, sorted=False)[1]
    masked_scores = utils.create_grouped_scores(scores, group_idx, num_nodes)

    topk_idx = paddle.topk(masked_scores, num_topk, axis=-1, largest=True, sorted=False)[1]
    topk_weights = paddle.ones(shape=[num_tokens, num_topk], dtype=paddle.float32) * rank
    topk_weights_pure_rand = paddle.randn(shape=[num_tokens, num_topk], dtype=paddle.float32)

    if dump_input:
        utils.dump(x, 'x', rank)
        utils.dump(x_pure_rand, 'x_pure_rand', rank)
        utils.dump(x_e4m3, 'x_e4m3', rank)

        utils.dump(topk_idx, 'topk_idx', rank)
        utils.dump(topk_weights, 'topk_weights', rank)
        utils.dump(topk_weights_pure_rand, 'topk_weights_pure_rand', rank)

    return x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand


def load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts):
    # x = utils.load("x", local_rank)
    # x_pure_rand = utils.load("x_pure_rand", local_rank)
    # #x_e4m3 = utils.load("x_e4m3", local_rank, "tuple")

    # topk_idx = utils.load("topk_idx", local_rank)
    # topk_weights = utils.load("topk_weights", local_rank)
    # topk_weights_pure_rand = utils.load("topk_weights_pure_rand", local_rank)

    def _load_tensor(rank, name, idx, typehint="tensor"):
        dump_dir = "/root/paddlejob/workspace/env_run/liuyiqun/outputs/ds_8nodes"
        filename = f"{dump_dir}/{idx}_{name}_rank{rank}.npy"
        if typehint == "tensor":
            x_np = np.load(filename)
            if x_np.dtype == np.uint16:
                x = paddle.to_tensor(x_np).view(paddle.bfloat16)
            elif x_np.dtype in [np.float32, np.int32, np.int64, np.int8]:
                x = paddle.to_tensor(x_np)
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


def test_main(local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, group: Group, use_random_input, dump_input):
    # Settings
    num_tokens = 4096
    hidden = 7168
    num_topk_groups = min(num_nodes, 4)
    num_topk = 8
    num_experts = (256 // num_ranks) * num_ranks

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    if profile:
        profile_paddle.push_record_event("init_input_tensors")
    if use_random_input:
        x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand = init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input)
    else:
        input_tensors = load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts)
    if profile:
        profile_paddle.pop_record_event()

    #paddle.distributed.barrier()

    # test bfloat16
    buffer = fused_a2a.get_buffer(group, hidden * 2)

    #paddle.distributed.barrier()

    num_warmups = 100
    num_tests = 1000

    start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    for i in range(num_warmups + num_tests + 1):
        #if i == 1:
        #    paddle.distributed.barrier()

        if profile:
            profile_paddle.push_record_event(f"test_{i}")

        if not use_random_input:
            inputs_i = input_tensors[i % 20]
            x = inputs_i["x"]
            topk_idx = inputs_i["topk_idx"]
            topk_weights = inputs_i["topk_weights"]

        if i == num_warmups:
            paddle.distributed.barrier(group)
            paddle.device.synchronize()
            cpu_start = time.time()

        if i >= num_warmups and i < num_warmups + num_tests:
            # Record
            batch_start = time.time()
            start_events[i - num_warmups].record()

        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, dispatch_event = fused_a2a.fused_dispatch_forward_func(
            x=x,
            token_indices=topk_idx,
            token_probs=topk_weights,
            num_experts=num_experts,
            group=group,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False
        )

        # combined_x, event = fused_a2a.fused_combine_forward_func(
        #     x=recv_x,
        #     handle=handle,
        #     previous_event=None,
        #     async_finish=False,
        #     allocate_on_comm_stream=False
        # )

        if i >= num_warmups and i < num_warmups + num_tests:
            end_events[i - num_warmups].record()

        if i < num_warmups + 20:
            random_time = random.uniform(0.02, 0.08)
            time.sleep(random_time)
        else:
            time.sleep(0.02)

        if profile:
            profile_paddle.pop_record_event()
        if i >= num_warmups and i < num_warmups + num_tests:
            batch_cpu_time = time.time() - batch_start
            #if local_rank == 0:
            #   print(f"-- {i - num_warmups}-th running, cpu_time: {batch_cpu_time:.5f} s")

    paddle.distributed.barrier(group)
    paddle.device.synchronize()

    cpu_runtime = time.time() - cpu_start
    avg_cpu_time = cpu_runtime / num_tests

    gpu_times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    avg_gpu_time = np.average(gpu_times)
    max_gpu_time = np.max(gpu_times)
    min_gpu_time = np.min(gpu_times)

    print(f"-- rank: {rank}, avg_cpu_time: {avg_cpu_time:.5f} s; gpu_time: avg={avg_gpu_time:.5f} s, max={max_gpu_time:.5f} s, min={min_gpu_time:.5f} s")

    paddle.distributed.barrier(group)
    paddle.device.synchronize()

    avg_cpu_time_all_ranks = []
    avg_gpu_time_all_ranks = []
    max_gpu_time_all_ranks = []
    min_gpu_time_all_ranks = []
    dist.all_gather_object(avg_cpu_time_all_ranks, avg_cpu_time, group=group)
    dist.all_gather_object(avg_gpu_time_all_ranks, avg_gpu_time, group=group)
    dist.all_gather_object(max_gpu_time_all_ranks, max_gpu_time, group=group)
    dist.all_gather_object(min_gpu_time_all_ranks, min_gpu_time, group=group)
    if rank == 0:
        avg_cpu_time = np.average(np.array(avg_cpu_time_all_ranks))
        avg_gpu_time = np.average(np.array(avg_gpu_time_all_ranks))
        max_gpu_time = np.average(np.array(max_gpu_time_all_ranks))
        min_gpu_time = np.average(np.array(min_gpu_time_all_ranks))
        print(f"-- avg_cpu_time_of_all_ranks: {avg_cpu_time:.5f} s; gpu_time_of_all_ranks: avg={avg_gpu_time:.5f} s, max={max_gpu_time:.5f} s, min={min_gpu_time:.5f} s")


def test_loop(num_local_ranks: int):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()

    num_ranks = dist.get_world_size(group)
    rank = dist.get_rank(group)

    num_nodes = int(num_ranks / 8)
    local_rank = rank % 8
    print(f'local_rank:{local_rank}, num_local_ranks:{num_local_ranks}, num_ranks:{num_ranks}, rank:{rank}')

    assert num_local_ranks == 8 and num_ranks > 8
    paddle.seed(rank)

    use_random_input = False
    dump_input = False

    print(f"-- profile: {profile}")
    if profile:
        profile_paddle.switch_profile(0, 0, 1)

    test_main(local_rank, num_local_ranks, num_ranks, num_nodes, rank, group, use_random_input, dump_input)

    if profile:
        profile_paddle.switch_profile(1, 0, 1)

if __name__ == '__main__':
    num_processes = 8
    #torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
    world_size = int(os.getenv('WORLD_SIZE', 1))
    mp_degree = world_size * num_processes
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "mp_degree": mp_degree,
    }
    fleet.init(is_collective=True, strategy=strategy)
    test_loop(num_processes)
