python3 run_mae.py \
  --dataset_name ~/NUS-NeuralNetworks \
  --train_dir "data/mae_train/*" \
  --validation_dir "data/mae_test/*" \
  --output_dir ~/NUS-NeuralNetworks/models/MAE_full100 \
  --remove_unused_columns False \
  --label_names pixel_values \
  --mask_ratio 0.5 \
  --base_learning_rate 1.5e-4 \
  --lr_scheduler_type cosine \
  --weight_decay 0.05 \
  --num_train_epochs 100 \
  --warmup_ratio 0.05 \
  --per_device_train_batch_size 48 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 12 \
  --logging_strategy epoch \
  --logging_steps 200 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 1337 \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --token None