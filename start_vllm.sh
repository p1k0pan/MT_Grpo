#!/bin/bash

# 激活 Conda 环境
echo "🔄 正在切换到 Conda 环境 pjh_grpo_mt..."
eval "$(conda shell.bash hook)"
conda activate pjh_grpo_mt

# 检查 conda 环境是否激活成功
if [[ "$CONDA_DEFAULT_ENV" == "pjh_grpo_mt" ]]; then
  echo "✅ Conda 环境 pjh_grpo_mt 已成功激活！"
else
  echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
  exit 1
fi

model_name=newest_model_name_here  # Replace with your actual model name
# model_name=/mnt/data/users/liamding/data/models/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0,1 vllm serve ${model_name} --dtype bfloat16 --gpu-memory-utilization 0.9 --tensor-parallel-size 2