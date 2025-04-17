#!/bin/bash

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.8/torch_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

export PYTHONPATH=${WORK_ROOT}/PaPerf:$PYTHONPATH

#export NVSHMEM_DIR=$ROOT_DIR/third-party/nvshmem
#export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"

START_RANK=46
END_RANK=54

if [[ ${PADDLE_TRAINER_ID} -lt $START_RANK ]]; then
    exit 0
fi

if [[ ${PADDLE_TRAINER_ID} -ge $END_RANK ]]; then
    exit 0
fi

rank=$(($PADDLE_TRAINER_ID - $START_RANK))
nnodes=$(($END_RANK - $START_RANK))
echo "rank: ${rank}, nnodes: ${nnodes}"

python -c "import torch; print(torch.__version__)"

#master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
export MASTER_ADDR="10.95.238.87" # 46
#master="10.95.238.99"  # 48
#master="10.95.237.154" # 32
#master="10.95.244.212"  # 8
export MASTER_PORT=8367
export WORLD_SIZE=$nnodes
export RANK=$rank

export NCCL_DEBUG=WARN
#export NVSHMEM_DEBUG=DEBUG
#export NVSHMEM_DEBUG=TRACE

# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8 
#export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# 开启AR功能
#export NCCL_IB_ADAPTIVE_ROUTING=1

export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162

#export NVSHMEM_IB_ENABLE_IBGDA=true
#export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=xgbe0
#export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET

#export NVSHMEM_DEBUG=INFO

export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_simple_kernel_${WORLD_SIZE}.torch"
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx --nic-metrics=true --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_internode_${WORLD_SIZE}nodes_rank${RANK}.torch"

rm -rf core.*

${nsys_args} python test_internode_latency.py
