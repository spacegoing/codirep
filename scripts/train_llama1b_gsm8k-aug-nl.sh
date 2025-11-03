#!/usr/bin/env bash
set -xeuo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/run_${TIMESTAMP}.log"
exec &> >(tee -a "$LOGFILE")
echo "Logging all output to: $LOGFILE"

SAVE_DIR=~/codi_ckpt/codi_nl_llama

mkdir -p "$SAVE_DIR"

# cp scripts/train_28.20_ce_llama1b_dynamic-teacher_factor-exp_lat6.sh "$SAVE_DIR"


export http_proxy="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export https_proxy="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export HTTP_PROXY="http://jdtcom:709a64b73eb3@10.119.176.202:3128"
export HTTPS_PROXY="http://jdtcom:709a64b73eb3@10.119.176.202:3128"

# export NCCL_DEBUG=INFO

export NCCL_IB_GID_INDEX=5
export NCCL_IB_TC=138
export NCCL_IB_QPS_CONNECTION=8
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=32
export NCCL_RUNTIME_CONNECT=0
export NCCL_NVLS_ENABLE=1

export OMPI_MCA_btl_tcp_if_include=eth0

export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_IB_TRAFFIC_CLASS=138
export NVSHMEM_IB_ENABLE_IBGDA=1

export CUDA_DEVICE_MAX_CONNECTIONS=1

# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000

# export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# export NVTE_FUSED_ATTN=0
# export NVTE_FLASH_ATTN=1
# export NVTE_NORM_FWD_USE_CUDNN=1
# export NVTE_NORM_BWD_USE_CUDNN=1
# export NVTE_EXT_MARGIN_SM=20
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=2

# export CUDNN_LOGERR_DBG=1
# export CUDNN_LOGDEST_DBG=stderr

export NNODES=1
export WORLD_SIZE=1
export RANK=0
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12453
export GPU_NUM=8
export GPUS_PER_NODE=8

python -u train.py \
	--output_dir "$SAVE_DIR" \
  --expt_name gsm8k_llama1b_latent_baseline \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path /public/lichang93/stCodeLab/downloads/models/Llama-3.2-1B-Instruct \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
	--bf16 \
	--num_train_epochs 3 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_total_limit 1 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
  --num_latent 6 \
  --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 1000 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 256
