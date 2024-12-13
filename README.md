# Iterative Ensemble Training with Anti-Gradient Control for Mitigating Memorization in Diffusion Models

This repo contains code about text-conditioned generation for this papers.


[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73021-4_7)
[arxiv](https://arxiv.org/abs/2407.15328)

The main implementation（unconditional generation） and detail introduction in [here](https://github.com/liuxiao-guan/IET_AGC).

## Set up

Install conda environment

```
conda env create -f env.yaml
conda activate diffrep
```
We used RTX-4090 machines to train the models. For inference or to compute the metrics, smaller machines will do.

## Finetuning a model

```
python src/diff_train_fed_acce_es.py \
  --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
  --instance_data_dir <training_data_path> \
  --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
  --learning_rate=2.5e-6 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps=5000  --max_train_steps=2000 \
  --train_batch_size=8 --save_steps=2000 --modelsavesteps 40000 --duplication nodup  \
  --output_dir=<path_to_save_model> --class_prompt instancelevel_blip --instance_prompt_loc <path_to_captions_json> \
  --clients_num=4 --total_round=25 --modelsaverounds=5 --trainggpu=4 --mem_ratio=1.25

```
-  To train a model, you need the same number of RTX-4090 as clients_num.



## Inference from a finetuned model

```
python diff_inference.py --modelpath <path_to_finetuned_model> -nb <number_of_inference_generations>
```

## Computing metrics

This script computes similairity scores, fid scores and a few other metrics. Logged to `wandb`.

```
python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct --pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --query_dir <path_to_generated_data> --val_dir <path_to_training_data>

```
You may want to download the SSCD checkpoint first [here](https://drive.google.com/file/d/1PAMwyK5b5zi6WBvyENtWuWr0lpT-TYMk/view?usp=sharing)

## Data

Download the LAION-10k split [here](https://drive.google.com/drive/folders/1TT1x1yT2B-mZNXuQPg7gqAhxN_fWCD__?usp=sharing).

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.



## Cite us

```
@inproceedings{liu2025iterative,
  title={Iterative ensemble training with anti-gradient control for mitigating memorization in diffusion models},
  author={Liu, Xiao and Guan, Xiaoliu and Wu, Yu and Miao, Jiaxu},
  booktitle={European Conference on Computer Vision},
  pages={108--123},
  year={2025},
  organization={Springer}
}
```

## Acknowledgement
We would like to thank the authors of previous related projects for generously sharing their code, especially the [Somepail](https://github.com/somepago/DCR), from which our code is adapted and [Wen](https://github.com/YuxinWenRick/diffusion_memorization) who provides us the SSCD checkpoint.
