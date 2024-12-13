
# echo "Start the CWR: rand_word_repeat"
# python diff_inference.py  -nb 4000 --dataset laion --capstyle instancelevel_blip --rand_augs rand_word_repeat
# echo "Start the RT: rand_word_add"
# python diff_inference.py  -nb 4000 --dataset laion --capstyle instancelevel_blip --rand_augs rand_word_add
# echo "Start the RNA: rand_numb_add"
# python diff_inference.py  -nb 4000 --dataset laion --capstyle instancelevel_blip --rand_augs rand_numb_add
# echo "Start the GN: rand_noise_lam 0.1"
# python diff_inference.py  -nb 4000 --dataset laion --capstyle instancelevel_blip --rand_noise_lam 0.1 


# CUDA_VISIBLE_DEVICES=0 python diff_inference.py  -nb 4 --dataset laion --capstyle instancelevel_blip --modelpath /root/autodl-tmp/logs/Projects/DCR/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu8_mr1.25/ --iternum=25
CUDA_VISIBLE_DEVICES=0 python diff_inference.py  --modelpath /root/autodl-tmp/logs/Projects/DCR/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu8_mr1.25 -nb 4000 --dataset laion --capstyle instancelevel_blip --iternum=25
# CUDA_VISIBLE_DEVICES=0 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
# --pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
# --query_dir /root/autodl-tmp/logs/Projects/DCR/inferences/defaultsd/laion --val_dir /root/autodl-tmp/laion_10k/train/
