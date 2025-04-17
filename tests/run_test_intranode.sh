#!/bin/bash

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.8/torch_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

export MASTER_ADDR=10.54.98.83
export MASTER_PORT=8364
export WORLD_SIZE=1
export RANK=$PADDLE_TRAINER_ID

python test_intranode.py
