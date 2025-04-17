import os
import sys
import time

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle.distributed.communication.deep_ep as deep_ep
from paddle.distributed.communication.group import Group
from paddle.base.core import Config
import numpy as np

# noinspection PyUnresolvedReferences
import utils
from utils import bench, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
#import test_low_latency
try:
    from paperf import profile_paddle
    has_paperf = True
except ImportError:
    has_paperf = False


def test_main(num_sms: int, local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: Group, use_random_input: bool, dump_input: bool, dump_output: bool, tune_performance: bool):
    # Settings
    num_tokens, hidden, num_topk_groups, num_topk, num_experts = 4096, 7168, min(num_nodes, 4), 8, (256 // num_ranks) * num_ranks
    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    if use_random_input:
        # Random data
        x = paddle.ones(shape=[num_tokens, hidden], dtype=paddle.bfloat16) * rank
        x_pure_rand = paddle.randn(shape=[num_tokens, hidden], dtype=paddle.bfloat16)
        x_e4m3 = per_token_cast_to_fp8(x)
        scores = paddle.randn(shape=[num_tokens, num_experts], dtype=paddle.float32).abs() + 1

        group_scores = scores.view([num_tokens, num_nodes, -1]).amax(axis=-1)
        group_idx = paddle.topk(group_scores, num_topk_groups, axis=-1, sorted=False)[1]
        masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
        topk_idx = paddle.topk(masked_scores, num_topk, axis=-1, largest=True, sorted=False)[1]

        topk_weights = paddle.ones(shape=[num_tokens, num_topk], dtype=paddle.float32) * rank
        topk_weights_pure_rand = paddle.randn(shape=[num_tokens, num_topk], dtype=paddle.float32)

        if dump_input:
            utils.dump(topk_idx, "topk_idx", local_rank)
    else:
        x = utils.load("x", local_rank)
        x_pure_rand = utils.load("x_pure_rand", local_rank)
        x_e4m3 = utils.load("x_e4m3", local_rank, "tuple")

        topk_idx = utils.load("topk_idx", local_rank)
        topk_weights = utils.load("topk_weights", local_rank)
        topk_weights_pure_rand = utils.load("topk_weights_pure_rand", local_rank)

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = paddle.not_equal(rdma_idx, paddle.full_like(rdma_idx, -1)).sum().item()

    current_node = rank // num_local_ranks
    mask_rdma_only = (rdma_idx != current_node) & (rdma_idx != -1)
    num_rdma_only_token_sent = mask_rdma_only.sum().item()
    print(f"-- [local_rank={local_rank}, rank={rank}] num_rdma_token_sent: {num_rdma_token_sent}, num_rdma_token_sent_rdma_only: {num_rdma_only_token_sent}")

    if use_random_input:
        # Expert meta
        num_tokens_per_expert = paddle.zeros(shape=[num_experts, ], dtype=paddle.int32)
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
        gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
        dist.all_reduce(gbl_num_tokens_per_expert, group=group)

        # Rank layout meta
        num_tokens_per_rank = paddle.empty([num_ranks, ], dtype=paddle.int32)
        num_tokens_per_rdma_rank = paddle.empty([num_nodes, ], dtype=paddle.int32)
        token_idx_in_rank = paddle.full([num_ranks, num_tokens], -1, dtype=paddle.int64)
        for i in range(num_ranks):
            num_tokens_per_rank[i] = (rank_idx == i).sum()
            token_sel = (rank_idx == i).cast(paddle.int32).max(axis=-1)
            count = token_sel.sum().item()
            tokens = paddle.argsort(token_sel.cast(paddle.int32), descending=True)
            tokens[:count] = paddle.sort(tokens[:count])
            token_idx_in_rank[i][tokens[:count]] = paddle.arange(count, dtype=paddle.int64)
        for i in range(num_nodes):
            num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
        token_idx_in_rank = token_idx_in_rank.t().contiguous().cast(paddle.int32)
        is_token_in_rank = token_idx_in_rank >= 0
        gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
        dist.all_reduce(gbl_num_tokens_per_rank, group=group)
    else:
        num_tokens_per_rank = utils.load('num_tokens_per_rank', local_rank)
        num_tokens_per_rdma_rank = utils.load('num_tokens_per_rdma_rank', local_rank)
        is_token_in_rank = utils.load('is_token_in_rank', local_rank)
        num_tokens_per_expert = utils.load('num_tokens_per_expert', local_rank)
        gbl_num_tokens_per_rank = utils.load('gbl_num_tokens_per_rank', local_rank)
        gbl_num_tokens_per_expert = utils.load('gbl_num_tokens_per_expert', local_rank)

    ############################################################################################################
    # get_dispatch_layout
    ############################################################################################################

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    if dump_output:
        utils.dump(ref_num_tokens_per_rank, 'ref_num_tokens_per_rank', local_rank)
        utils.dump(ref_num_tokens_per_rdma_rank, 'ref_num_tokens_per_rdma_rank', local_rank)
        utils.dump(ref_num_tokens_per_expert, 'ref_num_tokens_per_expert', local_rank)
        utils.dump(ref_is_token_in_rank, 'ref_is_token_in_rank', local_rank)

    assert paddle.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert paddle.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert paddle.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert paddle.allclose(ref_is_token_in_rank, is_token_in_rank)

    t = bench(group, lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print()
    paddle.distributed.barrier(group)
    time.sleep(1)

    ############################################################################################################

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert paddle.allclose(check_x.amin(axis=1), check_x.amax(axis=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    dtype_str = "FP8" if isinstance(current_x, tuple) else "BF16"
                    dump_prefix = f'{dtype_str}_{"with" if with_topk else "without"}_top-k_async_{async_mode}_previous_{previous_mode}_'
                    if local_rank == 0:
                        print(f'[testing] Running with {dtype_str}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='\n')


                    dispatch_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'config': config,
                        'async_finish': async_mode
                    }
                    if with_topk:
                        dispatch_args.update({
                            'topk_idx': topk_idx,
                            'topk_weights': topk_weights_pure_rand if not isinstance(current_x, tuple) else topk_weights
                        })

                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})

                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()

                    if dump_output:
                        utils.dump(recv_x, f'{dump_prefix}recv_x', local_rank)
                        utils.dump(recv_topk_idx, f'{dump_prefix}recv_topk_idx', local_rank)
                        utils.dump(recv_topk_weights, f'{dump_prefix}recv_topk_weights', local_rank)
                        utils.dump(recv_num_tokens_per_expert_list, f'{dump_prefix}recv_num_tokens_per_expert_list', local_rank)

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-4]

                    if dump_output:
                        utils.dump(recv_gbl_rank_prefix_sum, f"{dump_prefix}recv_gbl_rank_prefix_sum", local_rank)

                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.shape[0], f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.shape[0]}'
                    assert gbl_num_tokens_per_expert.view([num_ranks, -1])[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        pass
                        # check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.equal(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.equal(i).sum().item() == count

                        if use_random_input:
                            # Check `topk_weights`
                            if current_x is not x_pure_rand:
                                recv_topk_weights[recv_topk_idx.equal(-1)] = recv_topk_weights.amax(axis=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.equal(-1)]
                                # check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    # NOTES: handle must be refreshed
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})

                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()

                        if dump_output:
                            utils.dump(recv_x, f'{dump_prefix}recv_x_wo_topk', local_rank)

                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                        if use_random_input:
                            if current_x is not x_pure_rand:
                                pass
                                # check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    if not use_random_input:
                        recv_x = utils.load(f"{dump_prefix}recv_x_combine_input", local_rank)

                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}

                    if with_topk:
                        if not use_random_input:
                            recv_topk_weights = utils.load(f"{dump_prefix}recv_topk_weights_input", local_rank)
                        combine_args.update({'topk_weights': recv_topk_weights})

                    if dump_input:
                        utils.dump(recv_x, f'{dump_prefix}recv_x_combine_input', local_rank)
                        if with_topk:
                            utils.dump(recv_topk_weights, f'{dump_prefix}recv_topk_weights_input', local_rank)
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})

                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()

                    if dump_output:
                        utils.dump(combined_x, f"{dump_prefix}combined_x", local_rank)
                        utils.dump(combined_topk_weights, f"{dump_prefix}combined_topk_weights", local_rank)

                    # check_x = combined_x.cast(paddle.float32) / is_token_in_rank.sum(axis=1).unsqueeze(1)
                    if use_random_input:
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        # assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        pass
                        # check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        # ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        # assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_rdma_only_send_bytes = num_rdma_only_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes
                    combine_bf16_rdma_only_recv_bytes = dispatch_bf16_rdma_only_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)

    if local_rank == 0:
        print()

    def print_tensor_info(t, name):
        print(f"-- {name}: data_ptr={t.data_ptr()}, shape={t.shape}, dtype={t.dtype}") 

    profile = False
    profile = profile and has_paperf

    if profile:
        profile_paddle.switch_profile(0, 0, 1)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        dtype_str = "FP8" if isinstance(current_x, tuple) else "BF16"
        if profile:
            profile_paddle.push_record_event(f"Tune_Dispatch_{dtype_str}")

        best_time, best_cpu_time, best_results = 1e10, 1e10, None

        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        rdma_only_send_bytes = (dispatch_bf16_rdma_only_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_only_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes

        for nvl_chunk_size in range(4, 33, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
                if profile:
                    profile_paddle.push_record_event(f"Dispatch_{dtype_str}_Config({config_str})")

                config = Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                result_times = bench(group, lambda: buffer.dispatch(**tune_args))
                t = result_times[0]
                cpu_t = result_times[3]

                if profile:
                    profile_paddle.pop_record_event()

                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                    best_cpu_time = cpu_t

                if local_rank == 0:
                    rdma_send_GBs = rdma_send_bytes / 1e9 / t
                    rdma_only_send_GBs = rdma_only_send_bytes / 1e9 / t
                    nvl_recv_GBs = nvl_recv_bytes / 1e9 / t
                    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {rdma_send_GBs:.2f} GB/s (RDMA + NVL), {rdma_only_send_GBs:.2f} GB/s (RDMA), {nvl_recv_GBs:.2f} GB/s (NVL) (time: {t:.5f} s, cpu_time: {cpu_t:.5f} s)')

        if profile:
            profile_paddle.pop_record_event()

        if local_rank == 0:
            rdma_send_GBs = rdma_send_bytes / 1e9 / best_time
            rdma_only_send_GBs = rdma_only_send_bytes / 1e9 / best_time
            nvl_recv_GBs = nvl_recv_bytes / 1e9 / best_time
            print(f'[tuning] Best dispatch ({dtype_str}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {rdma_send_GBs:.2f} GB/s (RDMA + NVL), {rdma_only_send_GBs:.2f} (RDMA), {nvl_recv_GBs:.2f} GB/s (NVL) (time: {best_time:.5f} s, cpu_time: {best_cpu_time:.5f} s)')
            print()

        if isinstance(current_x, tuple):
            if profile:
                profile_paddle.push_record_event("Gather_Best_Config")

            # Gather FP8 the best config from rank 0
            best_dispatch_results = paddle.to_tensor([best_results[0], best_results[1], best_results[2]], dtype=paddle.int32)
            all_best_fp8_results_list = [paddle.zeros_like(best_dispatch_results) for _ in range(paddle.distributed.get_world_size(group))]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()

            if profile:
                profile_paddle.pop_record_event()

    paddle.distributed.barrier(group)
    print(f"========================================================================")

    config_str = f"sms={best_dispatch_results[0]},nvl={best_dispatch_results[1]},{nvl_buffer_size},rdma={best_dispatch_results[2]},{rdma_buffer_size}"
    if profile:
        profile_paddle.push_record_event(f"Best_Dispatch_BF16_Config({config_str})")

    dispatch_config = Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)
    #dispatch_config = Config(24, 20, 512, 32, 128)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}

    #if local_rank == 0:
    #    print_tensor_info(x, "x")
    #    print_tensor_info(num_tokens_per_rank, "num_tokens_per_rank")
    #    print_tensor_info(num_tokens_per_rdma_rank, "num_tokens_per_rdma_rank")
    #    print_tensor_info(is_token_in_rank, "is_token_in_rank")
    #    print_tensor_info(num_tokens_per_expert, "num_tokens_per_expert")
    #    print(f"-- dispatch_args: {dispatch_args}")
    #    print(f"-- dispatch_config: {best_dispatch_results[0]}, {best_dispatch_results[1]}, {nvl_buffer_size}, {best_dispatch_results[2]}, {rdma_buffer_size}")

    for i in range(1):
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    if profile:
        profile_paddle.pop_record_event()

    #if local_rank == 0:
    #    print_tensor_info(recv_x, "recv_x")

    if profile:
        profile_paddle.push_record_event(f"Tune_Combine_BF16")

    # Tune combine performance
    best_time, best_cpu_time, best_results = 1e10, 1e10, None
    for nvl_chunk_size in range(1, 5, 1):
        for rdma_chunk_size in range(8, 33, 4):
            config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
            if profile:
                profile_paddle.push_record_event(f"Combine_BF16_Config({config_str})")

            config = Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            result_times = bench(group, lambda: buffer.combine(**tune_args))
            t = result_times[0]
            cpu_t = result_times[3]

            if profile:
                profile_paddle.pop_record_event()

            if local_rank == 0:
                combine_bf16_rdma_recv_GBs = combine_bf16_rdma_recv_bytes / 1e9 / t
                combine_bf16_rdma_only_recv_GBs = combine_bf16_rdma_only_recv_bytes / 1e9 / t
                combine_bf16_nvl_send_GBs = combine_bf16_nvl_send_bytes / 1e9 / t
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {combine_bf16_rdma_recv_GBs:.2f} GB/s (RDMA + NVL), {combine_bf16_rdma_only_recv_GBs:.2f} GB/s (RDMA), {combine_bf16_nvl_send_GBs:.2f} GB/s (NVL) (time: {t:.5f} s, cpu_time: {cpu_t:.5f} s)')
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                    best_cpu_time = cpu_t

    if profile:
        profile_paddle.pop_record_event()

    if profile:
        profile_paddle.switch_profile(1, 0, 1)

    if local_rank == 0:
        combine_bf16_rdma_recv_GBs = combine_bf16_rdma_recv_bytes / 1e9 / best_time
        combine_bf16_rdma_only_recv_GBs = combine_bf16_rdma_only_recv_bytes / 1e9 / best_time
        combine_bf16_nvl_send_GBs = combine_bf16_nvl_send_bytes / 1e9 / best_time
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {combine_bf16_rdma_recv_GBs:.2f} GB/s (RDMA + NVL), {combine_bf16_rdma_only_recv_GBs:.2f} GB/s (RDMA), {combine_bf16_nvl_send_GBs:.2f} GB/s (NVL) (time: {best_time:.5f} s, cpu_time: {best_cpu_time:.5f} s)')
        print()


# noinspection PyUnboundLocalVariable
def test_loop(num_local_ranks):
    # Please make sure AR (Adaptive Routing) is turned off when running normal internode kernels,
    # rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility = False
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    hcg = fleet.get_hybrid_communicate_group()
    ep_group = hcg.get_model_parallel_group()

    buffer = deep_ep.Buffer(ep_group, int(1e9), int(1e9), low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1))

    num_ranks = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)

    num_nodes = int(num_ranks / 8)
    local_rank = rank % 8
    print(f'local_rank:{local_rank}, num_local_ranks:{num_local_ranks}, num_ranks:{num_ranks}, rank:{rank}')

    assert num_local_ranks == 8 and num_ranks > 8
    paddle.seed(rank)

    use_random_input = True
    dump_input = False
    dump_output = False
    tune_performance = True

    for i in (24, ):
        test_main(i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, ep_group, use_random_input, dump_input, dump_output, tune_performance)
        if local_rank == 0:
            print()

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)


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
