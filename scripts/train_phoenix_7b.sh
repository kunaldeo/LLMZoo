model_name_or_path=bigscience/bloomz-7b1-mt
model_max_length=2048
data_path=data/data.json
valid_data_path=data/test.json
output_dir=checkpoints/phoenix_7b/

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=12375 \
  train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --valid_data_path ${valid_data_path} \
  --output_dir ${output_dir} \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "epoch" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'BloomBlock' \
  --tf32 True \
  --gradient_checkpointing True
