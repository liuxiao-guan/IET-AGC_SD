
python diff_train.py \
  --pretrained_model_name_or_path /root/autodl-tmp/logs/Projects/DCR/train/_instancelevel_blip_nodup_bs4_gpu4/checkpoint/ \
  --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
  --resolution=256 --gradient_accumulation_steps=4 --center_crop --random_flip \
  --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps=10000  --max_train_steps=100000 \
  --train_batch_size=4 --save_steps=4000 --modelsavesteps 10000 --duplication nodup  \
  --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_test/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \



# python diff_train.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
#   --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=5000  --max_train_steps=2 \
#   --train_batch_size=4 --save_steps=2 --modelsavesteps 2 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_test/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \

