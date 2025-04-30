#!/bin/bash


set -euo pipefail

TS=`date +%Y_%m_%d_%H_%M_%S`

DATA_PATH="/root/dataset/enwiki-100m_text_document"
TRAIN_ITERS=50

if [ $GQA == "1" ]; then
    GQA_ARGS="--group-query-attention --num-query-groups $NUM_QUERY_GROUPS --no-context-parallel-comm-overlap-gemm"
fi

TP_OVERLAP_ARGS="--overlap-sp-ag --overlap-sp-rs"

GPT_ARGS="
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    ${GQA_ARGS:-} \
    --micro-batch-size 1 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 1.5e-4 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 500000 \
    --lr-decay-style cosine \
    --min-lr 1.5e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --bf16 \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --swiglu \
    --rms-norm \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --hidden-dropout 0. \
    --attention-dropout 0. \
    --no-query-key-layer-scaling \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type NullTokenizer \
    --vocab-size 32004 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0
"

if [ -n "${HOSTFILE:-}" ]; then
    CLUSTER_MPI_ARGS="
        --hostfile $HOSTFILE \
        --mca plm_rsh_num_concurrent 600 \
        --mca routed_radix 600 \
        --mca btl_tcp_if_include ens99f3 \
        --mca oob_tcp_if_include ens99f3 \
        --mca btl_openib_allow_ib false \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_IB_QPS_PER_CONNECTION=8 \
        -x NCCL_IB_TIMEOUT=19 \
        -x NCCL_NET_OVERHEAD=1000 \
    "
fi

#        -x PYTHONPATH=../.. \
mkdir -p logs
set -x
NET_DEVICE="ens99f3"

   # --use-fast-rms-norm \
   # --use-fast-rope \
   mpirun --allow-run-as-root \
        ${CLUSTER_MPI_ARGS:-} \
        --mca btl self,tcp \
        --mca pml ob1 \
        --np $NUM_GPUS \
        --bind-to none --map-by slot \
        -x MPI_THREAD_SINGLE=1 \
        -x NCCL_DEBUG=WARN \
	-x GLOO_SOCKET_IFNAME=${NET_DEVICE} \
        -x NCCL_SOCKET_IFNAME=${NET_DEVICE} \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -x NVTE_FWD_LAYERNORM_SM_MARGIN=8 \
        -x NVTE_BWD_LAYERNORM_SM_MARGIN=8 \
        -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        -x PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21 \
        -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=6002 \
	-x NCCL_IB_GID_INDEX=3     \
	-x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/infrawaves/vccl_2.21.51-atc/build/lib \
    python3 ../pretrain_gpt.py \
    --use-distributed-optimizer \
    --num-layers-per-virtual-pipeline-stage $PP_l \
     --overlap-p2p-communication \
    --accumulate-allreduce-grads-in-fp32 \
    --initial-loss-scale 1 \
    --use-flash-attn \
    --tensor-model-parallel-size $TP \
    --sequence-parallel \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP \
    $TP_OVERLAP_ARGS \
    --kaimm-overlap-cp-slow-ctas 4 \
    --kaimm-cuda-synchronize-level 2 \
    --kaimm-gc-interval 9999 \
    --kaimm-async-dataloader \
    --prefetch-factor 64 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    2>&1 | tee logs/llama_$TS.txt
