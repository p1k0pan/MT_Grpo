# WANDB_MODE=offline \
    # --sleep_level 1 \
    # --offload_optimizer true \
    # --offload_model true \
    # --gc_collect_after_offload true \
WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data/mt-r1-zero-train.jsonl \
    --external_plugins ms-swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs comet \
    --reward_weights 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.5 \
    --log_completions true \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero3 \
    --report_to wandb

