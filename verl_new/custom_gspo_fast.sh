
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

# export WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c # set your wandb api key
export SWANLAB_API_KEY=57bftOCtg6exWFs81mtT1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# GSPO 特定参数设置
adv_estimator=grpo                  # 优势估计器
loss_mode=gspo                      # GSPO算法 (可选: gspo 或 gspo-token)
loss_agg_mode="seq-mean-token-mean" # GSPO推荐的损失聚合模式
clip_ratio_low=0.0003               # GSPO推荐的低剪切比率
clip_ratio_high=0.0004              # GSPO推荐的高剪切比率
max_response_length=$((1024 * 8))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
echo "🚀 启动 GSPO 训练，使用算法: ${loss_mode}"
echo "📊 剪切参数: clip_ratio_low=${clip_ratio_low}, clip_ratio_high=${clip_ratio_high}"
echo "🎯 优势估计器: ${adv_estimator}"
echo "📈 损失聚合模式: ${loss_agg_mode}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../data/train/parquet/train_base_enzh_zhen.parquet \
    data.val_files=../data/test/parquet/test_base_enzh_zhen.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/data/users/liamding/data/models/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32  \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
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
    custom_reward_function.path=comet_reward_batch.py \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.val_before_train=False \
    trainer.logger=['swanlab'] \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.project_name="verl-grpo" \
    trainer.experiment_name="qwen2.5_3b_gspo" \
    trainer.validation_data_dir=./checkpoints/qwen2.5_3b_gspo/validation_samples \
    trainer.log_val_generations=100 \
    trainer.total_epochs=1 $@ 2>&1 | tee custom_gspo_fast_3b.log
  

# trainer.project_name='verl_grpo_xwang' \
# trainer.experiment_name='qwen2.5_7b_r1-zero' \