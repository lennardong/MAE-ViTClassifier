python3 run_mae.py \
  --dataset_name ~/NUS-NeuralNetworks \
  --train_dir "data/mae_train/*" \
  --validation_dir "data/mae_test/*" \
  --output_dir ~/NUS-NeuralNetworks/models/MAE_full100_3 \
  --remove_unused_columns False \
  --label_names pixel_values \
  --mask_ratio 0.75 \
  --base_learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --num_train_epochs 100 \
  --warmup_ratio 0.05 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 12 \
  --logging_strategy epoch \
  --logging_steps 200 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 12 \
  --do_train \
  --overwrite_output_dir \
  # --do_eval False \
  # --max_train_samples 1000 \
