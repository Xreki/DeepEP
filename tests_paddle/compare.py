import os
import time
import numpy as np
from utils import init_dist, bench, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back


def load(name, local_rank, typehint="tensor"):
    if typehint=="tensor":
        x_np = np.load(f'{name}_rank{local_rank}.npy')
        return x_np
    elif typehint=="tuple":
        y_np = np.load(f'{name}_value_rank{local_rank}.npy')
        y_scale_np = np.load(f'{name}_scale_rank{local_rank}.npy')
        return (y_np, y_scale_np)
    else:
        assert False, f'invalid typehint: {typehint}'


def load_cmp(name, local_rank, typehint="tensor"):
  if typehint == "tensor":
      ref = load("torch_dump/"+name, local_rank, typehint)
      x = load("paddle_dump/"+name, local_rank, typehint)
      np.testing.assert_array_equal(x, ref, err_msg=f"{name} missmatch", strict=True)
  elif typehint == "tuple":
      ref1, ref2 = load("torch_dump/"+name, local_rank, typehint)
      x1, x2 = load("paddle_dump/"+name, local_rank, typehint)
      np.testing.assert_array_equal(x1, ref1, err_msg=f"{name} missmatch", strict=True)
      np.testing.assert_array_equal(x2, ref2, err_msg=f"{name} missmatch", strict=True)
  else:
      assert False, f'invalid typehint: {typehint}'


def test_main(local_rank: int):
    load_cmp("ref_num_tokens_per_rank", local_rank)
    load_cmp("ref_num_tokens_per_expert", local_rank)
    load_cmp("ref_is_token_in_rank", local_rank)

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x_type in ("hack", ("hack", "hack")):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x_type, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dump_prefix = f'{"FP8" if isinstance(current_x_type, tuple) else "BF16"}_{"with" if with_topk else "without"}_top-k_async_{async_mode}_previous_{previous_mode}_'

                    load_cmp(f"{dump_prefix}recv_x", local_rank, "tuple" if isinstance(current_x_type, tuple) else "tensor")
                    load_cmp(f"{dump_prefix}recv_topk_idx", local_rank)
                    load_cmp(f"{dump_prefix}recv_topk_weights", local_rank)
                    load_cmp(f"{dump_prefix}recv_num_tokens_per_expert_list", local_rank)

                    load_cmp(f"{dump_prefix}rank_prefix_matrix", local_rank)
                    if not with_topk:
                        load_cmp(f"{dump_prefix}recv_x_wo_topk", local_rank, "tuple" if isinstance(current_x_type, tuple) else "tensor")

                    load_cmp(f"{dump_prefix}combined_x", local_rank)
                    load_cmp(f"{dump_prefix}combined_topk_weights", local_rank)


if __name__ == '__main__':
    for i in range(8):
        test_main(i)
