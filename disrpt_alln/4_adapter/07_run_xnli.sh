# export XNLI_DIR=/path/to/XNLI

# python run_xnli.py \
#   --model_type bert \
#   --model_name_or_path bert-base-multilingual-cased \
#   --language de \
#   --train_language en \
#   --do_train \
#   --do_eval \
#   --data_dir $XNLI_DIR \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 2.0 \
#   --max_seq_length 128 \
#   --output_dir /tmp/debug_xnli/ \
#   --save_steps -1

rm -r xnli

python 07_xnli_datamaker.py \
  --model_name_or_path bert-base-german-cased \
  --language de \
  --train_language de \
  --train_dataset_task disrpt \
  --do_train \
  --do_eval \
  --do_predict \
  --per_gpu_train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 128 \
  --output_dir ./xnli_results2/ \
  --save_steps -1 \
  --overwrite_cache \
  --overwrite_output_dir \
  --max_train_samples 100
# --max_eval_samples 100 \
# --max_predict_samples 100


# python 07_xnli.py \
#   --model_name_or_path ./xnli_results \
#   --language de \
#   --train_language de \
#   --do_eval \
#   --do_predict \
#   --per_gpu_train_batch_size 64 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 4.0 \
#   --max_seq_length 128 \
#   --output_dir ./xnli_results/ \
#   --save_steps -1 \
#   --overwrite_cache