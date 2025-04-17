import os
import sys
import time
import numpy as np

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import alltoall
import utils
from utils import init_dist, create_grouped_scores

try:
    from paperf import profile_torch
    has_paperf = True
except ImportError:
    has_paperf = False


def print_tensor_info(t, name):
    #print(f"-- {name}: data_ptr={t.untyped_storage().data_ptr()}, shape={t.size()}, dtype={t.dtype}")
    print(f"-- {name}: shape={t.size()}, dtype={t.dtype}")


def test_main(local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, group: dist.ProcessGroup, use_random_input, dump_input):
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
        # Random data
        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
        x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        #x_e4m3 = per_token_cast_to_fp8(x)

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
            #utils.dump(x_e4m3, 'x_e4m3', local_rank)

            utils.dump(topk_idx, 'topk_idx', local_rank)
            utils.dump(topk_weights, 'topk_weights', local_rank)
            utils.dump(topk_weights_pure_rand, 'topk_weights_pure_rand', local_rank)
    else:
        x = utils.load("x", local_rank)
        x_pure_rand = utils.load("x_pure_rand", local_rank)
        #x_e4m3 = utils.load("x_e4m3", local_rank, "tuple")

        topk_idx = utils.load("topk_idx", local_rank)
        topk_weights = utils.load("topk_weights", local_rank)
        topk_weights_pure_rand = utils.load("topk_weights_pure_rand", local_rank)


    profile = False
    profile = profile and has_paperf

    # test bfloat16
    buffer = alltoall.get_buffer(group, alltoall.get_hidden_bytes(x))

    if profile:
        profile_torch.switch_profile(0, 0, 1)

    num_warmups = 100
    num_tests = 1000

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    for i in range(num_warmups + num_tests):
        if i == num_warmups:
            group.barrier()
            torch.cuda.synchronize()
            cpu_start = time.time()

        if i >= num_warmups:
            # Record
            batch_start = time.time()
            start_events[i - num_warmups].record()

            #group.barrier()
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, dispatch_event = alltoall.dispatch_forward(
                x=x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=num_experts,
                previous_event=None,
                async_finish=False,
                allocate_on_comm_stream=False
            )

            combined_x, event = alltoall.combine_forward(
                x=recv_x,
                handle=handle,
                previous_event=None,
                async_finish=False,
                allocate_on_comm_stream=False
            )

            end_events[i - num_warmups].record()
            batch_time = time.time() - batch_start
            if local_rank == 0:
                print(f"-- {i - num_warmups}-th running, cpu_time: {batch_time:.5f} s")
    torch.cuda.synchronize()
    group.barrier()

    cpu_runtime = time.time() - cpu_start
    avg_cpu_time = cpu_runtime / num_tests

    gpu_times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    avg_gpu_time = np.average(gpu_times)

    print(f"-- rank: {rank}, avg_cpu_time: {avg_cpu_time:.5f} s, avg_gpu_time: {avg_gpu_time:.5f} s")

    torch.cuda.synchronize()
    group.barrier()

    avg_cpu_time_all_ranks = [None, ] * num_ranks
    avg_gpu_time_all_ranks = [None, ] * num_ranks
    dist.all_gather_object(avg_cpu_time_all_ranks, avg_cpu_time, group=group)
    dist.all_gather_object(avg_gpu_time_all_ranks, avg_gpu_time, group=group)
    if rank == 0:
        avg_cpu_time = np.average(np.array(avg_cpu_time_all_ranks))
        avg_gpu_time = np.average(np.array(avg_gpu_time_all_ranks))
        print(f"-- avg_cpu_time_of_all_ranks: {avg_cpu_time:.5f} s, avg_gpu_time_of_all_ranks: {avg_gpu_time:.5f} s")


def test_loop(local_rank: int, num_local_ranks: int):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    use_random_input = True
    dump_input = False

    test_main(local_rank, num_local_ranks, num_ranks, num_nodes, rank, group, use_random_input, dump_input)


if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
