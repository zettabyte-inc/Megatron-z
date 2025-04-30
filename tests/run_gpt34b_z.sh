#!/bin/bash

set -euo pipefail

source ./gpt-34b

export SEQ_LENGTH=4096

export HOSTFILE=$1
NUM_NODE=$2
export MASTER_ADDR=$(cat $HOSTFILE | head -n 1 | sed -s 's/slots=8//g')
export NUM_GPUS=$((NUM_NODE*8))
export GLOBAL_BATCH_SIZE=$((NUM_NODE*8))

export TP=2
export CP=1
export PP=4
export PP_l=1

./pretrain_llama_z.sh
