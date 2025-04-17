#!/bin/bash

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.3/paddle_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

#nnodes=$PADDLE_TRAINERS_NUM
#rank=$PADDLE_TRAINER_ID
nnodes=1
rank=0

python -m paddle.distributed.launch --nnodes=$nnodes --rank $rank test_intranode.py
