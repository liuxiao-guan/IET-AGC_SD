CUDA_VISIBLE_DEVICES=1 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR/inferences_fed_es/laion_frozentext/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_time600_25 --val_dir /root/autodl-tmp/laion_10k/train/

