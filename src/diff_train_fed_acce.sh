CUDA_VISIBLE_DEVICES=1,2,3,4 python src/diff_train_fed_acce.py \
  --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
  --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
  --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
  --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps=5000  --max_train_steps=2000 \
  --train_batch_size=4 --save_steps=2000 --modelsavesteps 40000 --duplication nodup  \
  --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_fed/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \
  --clients_num=4 --total_round=50 --modelsaverounds=5 --trainggpu=4


CUDA_VISIBLE_DEVICES=2 python diff_inference.py  -nb 4000 --modelpath /root/autodl-tmp/logs/Projects/DCR/train_fed/_instancelevel_blip_nodup_bs4_gpu4/ --iternum=50

CUDA_VISIBLE_DEVICES=2 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR/inferences_fed/laion_frozentext/_instancelevel_blip_nodup_bs4_gpu4/ --val_dir /root/autodl-tmp/laion_10k/train/


  #python diff_inference.py  -nb 4000 --modelpath /root/autodl-tmp/logs/Projects/DCR/train_fed/_instancelevel_blip_nodup/ --iternum 100
#accelerate launch --mixed_precision=fp16
# python diff_train_fed.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
#   --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=5000  --max_train_steps=2 \
#   --train_batch_size=4 --save_steps=1 --modelsavesteps 2 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_fed/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \
#   --clients_num=2 --total_round=2 --modelsaverounds=2




