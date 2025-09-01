#!/bin/bash

set -x
# 激活 Conda 环境
# echo "🔄 正在切换到 Conda 环境 pjh_verl..."
# eval "$(conda shell.bash hook)"
# conda activate verl

# # 检查 conda 环境是否激活成功
# if [[ "$CONDA_DEFAULT_ENV" == "verl" ]]; then
#   echo "✅ Conda 环境 pjh_verl 已成功激活！"
# else
#   echo "❌ Conda 环境激活失败！当前环境为：$CONDA_DEFAULT_ENV"
#   exit 1
# fi

train_file_path=../data/train/parquet/train_base_enzh_zhen.parquet
test_file_path=../data/test/parquet/test_base_enzh_zhen.parquet
python3 ../data/process_data.py \
    --train_files "../data/train/json/train_zhen_6565.jsonl" "../data/train/json/train_enzh_6565.jsonl" \
    --test_files "../data/test/json/wmt23_zhen.jsonl" "../data/test/json/wmt24_enzh.jsonl" \
    --tokenizer_path /mnt/data/users/liamding/data/models/Qwen2.5-3B \
    --template_type "base" \
    --train_output_file ${train_file_path} \
    --test_output_file ${test_file_path}

# model_name=/mnt/data/users/liamding/data/models/Qwen2.5-14B-Instruct
# CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/data/users/liamding/data/models/Qwen2.5-14B-Instruct --dtype bfloat16 --gpu-memory-utilization 0.9 --tensor-parallel-size 1

# export WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c # set your wandb api key
export SWANLAB_API_KEY=57bftOCtg6exWFs81mtT1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# 确保使用新版本的verl代码
export PYTHONPATH=/mnt/data/users/liamding/data/MT_Grpo/verl_new:$PYTHONPATH

echo "🚀 启动分离损失模式的 GRPO 训练"
echo "📊 分离损失配置: translate部分=bleu+comet, think部分=thinking_score"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_separated_loss=True \
    data.train_files=../data/train/parquet/train_base_enzh_zhen.parquet \
    data.val_files=../data/test/parquet/test_base_enzh_zhen.parquet \
    data.train_batch_size=96 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/data/users/liamding/data/models/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    custom_reward_function.path=comet_reward_batch_llm_route.py \
    reward_model.reward_manager=batch \
    trainer.val_before_train=False \
    trainer.logger=['swanlab'] \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.project_name="verl-grpo-separated" \
    trainer.experiment_name="qwen2.5_3b_grpo_separated_loss" \
    trainer.validation_data_dir=./checkpoints/verl-grpo-separated/qwen2.5_3b_grpo_separated_loss/validation_samples \
    trainer.log_val_generations=100 \
    trainer.total_epochs=1 $@ 2>&1 | tee custom_grpo_separated_loss_3b.log

echo "✅ 分离损失模式 GRPO 训练完成"
echo "📊 监控指标包括: actor/pg_loss_tr, actor/pg_loss_th, actor/pg_clipfrac_tr, actor/pg_clipfrac_th"