import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
# Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer
    
    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency 
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = _buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream
    )
    
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(
    grad_recv_x: torch.Tensor,
    grad_recv_topk_weights: torch.Tensor,
    handle: Tuple,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle,
        topk_weights=grad_recv_topk_weights,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(
    x: torch.Tensor,
    handle: Tuple,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(
        x,
        handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(
    grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    handle: Tuple,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream
    )

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
