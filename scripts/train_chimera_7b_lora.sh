model_name_or_path=/data/llama-chimera-inst-chat-7b
model_max_length=2048
data_path=data/data.json
valid_data_path=data/test.json
output_dir=checkpoints/chimera_7b_lora/

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=12375 \
  train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --valid_data_path ${valid_data_path} \
  --output_dir ${output_dir} \
  --fp16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --save_strategy "steps" \
  --save_steps 100 \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_total_limit 3 \
  --learning_rate 3e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing False \
  --ddp_find_unused_parameters False \
  --lora True


