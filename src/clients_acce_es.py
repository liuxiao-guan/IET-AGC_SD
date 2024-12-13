from datasets import ObjectAttributeDataset, get_classnames
import argparse
import hashlib
import itertools
import math
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from datasets import ObjectAttributeDataset, get_classnames
from utils.draw_utils import concat_h
import copy
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]


    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

class Client(object):
    def __init__(self, client_id, train_dataset, train_dataloader):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.avg_loss = torch.zeros(1000)
        #self.device = device

        self.unet = None
        self.vae = None
        self.optimizer =None
        self.lr_scheduler = None
        self.text_encoder = None
        self.noise_scheduler = None
        self.device = None 

        self._step_cound = 0
    def init(self,unet,tokenizer,accelerator,args,logger,device):
        self.device = device 
        self.unet = copy.deepcopy(unet)
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters()) if args.train_text_encoder else self.unet.parameters()
        )
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        # import ipdb; ipdb.set_trace()
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load models and create wrapper for stable diffusion
        self.text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        self.vae.requires_grad_(False)
        if not args.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype
        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        # self.vae.to(accelerator.device, dtype=self.weight_dtype)
        # if not args.train_text_encoder:
        #     self.text_encoder.to(accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device)
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
       
        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def set_global_parameters(self,sum_parameters):
        self.unet.load_state_dict(sum_parameters, strict=True)
    def local_train(self,args,accelerator,OBJECTS,round,c):
        global_step = 0
        temp = list(self.train_dataset.prompts.values())
        choicelist = [x[0] for x in temp]
        # choicelist  = list(itertools.chain(*))
        if accelerator.is_main_process:
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(self.unet),
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                revision=args.revision, safety_checker=None,
            ).to(self.device)
            pipeline.set_use_memory_efficient_attention_xformers(True)
            genseed = args.generation_seed
            if args.class_prompt in ['instancelevel_blip','instancelevel_ogcap']: # and args.duplication == 'dup_image':
                # choicelist  = list(itertools.chain(*list(train_dataset.prompts.values())))
                rand_prompts = np.random.choice(choicelist,len(OBJECTS))

            if args.class_prompt in ['instancelevel_random']:
                
                temp = np.random.choice(choicelist,len(OBJECTS))
                # temp = np.random.choice(list(train_dataset.prompts.values()),len(OBJECTS))
                rand_prompts = []
                for p in temp:
                    rand_prompts.append(self.tokenizer.decode(eval(p)))
                print(rand_prompts)
                
            for count,object in enumerate(OBJECTS):
                if count > 2:
                    break
                if args.class_prompt == 'nolevel':
                    genseed+=1
                    prompt = f"An image"
                elif args.class_prompt == 'classlevel':
                    prompt = f"An image of {object}"
                elif args.class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
                    # import ipdb; ipdb.set_trace()
                    prompt = rand_prompts[count]
                    print('first one------------')
                    print(prompt)

                save_path = os.path.join(args.output_dir,"generations", f"{round}_{c}_{global_step:04d}_{object}.png")
                generator = torch.Generator(self.device).manual_seed(genseed)
                images = pipeline(prompt=prompt, height=args.resolution, width=args.resolution,
                                    num_inference_steps=50, num_images_per_prompt=4, generator=generator).images
                concat_h(*images, pad=4).save(save_path)
       # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0
        for epoch in range(args.num_train_epochs):
            self.unet.train()
            if args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):

                with accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    batch["pixel_values"] = batch["pixel_values"].to(self.device)
                    batch["input_ids"] = batch["input_ids"].to(self.device)
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    timesteps = timesteps.to(latents.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    # import ipdb; ipdb.set_trace()
                    encoder_hidden_states = self.text_encoder(batch["input_ids"])[0] 
                    if args.rand_noise_lam > 0:
                        encoder_hidden_states = encoder_hidden_states + args.rand_noise_lam*torch.randn_like(encoder_hidden_states)
                    if args.mixup_noise_lam > 0:
                        lam = np.random.beta(args.mixup_noise_lam, 1)
                        index = torch.randperm(encoder_hidden_states.shape[0]).cuda()
                        encoder_hidden_states = lam*encoder_hidden_states + (1-lam)*encoder_hidden_states[index,:]
                    # Predict the noise residual
                    model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    self.avg_loss = self.avg_loss.to(self.device)
                    #mem_ratio = 1.2
                    #arr_ema_decay = 0.8
                    ori_loss = torch.mean(loss, dim=(1, 2, 3))
                    ori_loss = ori_loss.to(self.device)
                    
                    mean_loss = self.avg_loss[timesteps]
                    mean_loss = mean_loss.to(self.device)
                    mask = ori_loss * args.mem_ratio >= mean_loss.float().detach().to(self.device)
                    #mask = torch.where((ori_loss * args.mem_ratio >= mean_loss.float()) | (timesteps >= 600), torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)).detach().to(latents.device)
                    tot = max(mask.sum().item(), 1)
                    loss = ((ori_loss * mask) * (latents.shape[0] / tot)).mean()
                    #index_array[indexes[mask == 0]] += 1
                    for index, timestep in enumerate(timesteps):
                        self.avg_loss[timestep] = args.arr_ema_decay * self.avg_loss[timestep] + (1. - args.arr_ema_decay) * ori_loss[
                            index].item()
                    
                    #loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                            if args.train_text_encoder
                            else self.unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            pipeline = DiffusionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                unet=accelerator.unwrap_model(self.unet),
                                text_encoder=accelerator.unwrap_model(self.text_encoder),
                                revision=args.revision, safety_checker=None,
                            ).to(self.device)
                            pipeline.set_use_memory_efficient_attention_xformers(True)
                            genseed = args.generation_seed
                            for count,object in enumerate(OBJECTS):
                                if count > 2:
                                    break
                                if args.class_prompt == 'nolevel':
                                    genseed+=1
                                    prompt = f"An image"
                                elif args.class_prompt == 'classlevel':
                                    prompt = f"An image of {object}"
                                elif args.class_prompt in ['instancelevel_blip','instancelevel_ogcap','instancelevel_random']:
                                    prompt = rand_prompts[count]
                                    print('_______in the loop______-')
                                    print(prompt)
                                    
                                save_path = os.path.join(args.output_dir,"generations", f"{round}_{c}_{global_step:04d}_{object}.png")
                                generator = torch.Generator(self.device).manual_seed(genseed)
                                images = pipeline(prompt=prompt, height=args.resolution, width=args.resolution,
                                                num_inference_steps=50, num_images_per_prompt=4,
                                                generator=generator).images
                                concat_h(*images, pad=4).save(save_path)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
                if accelerator.is_main_process and global_step % args.modelsavesteps == 0:
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(self.unet),
                        text_encoder=accelerator.unwrap_model(self.text_encoder),
                        revision=args.revision,
                    )
                    pipeline.save_pretrained(os.path.join(args.output_dir, f'checkpoint_{round}_{c}_{global_step}'))

            accelerator.wait_for_everyone()
        return self.unet.state_dict()

            


class ClientsGroup(object):

    def __init__(self,args,tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.clients_set = []
        self.data_allocation()
    def data_allocation(self):
        if self.args.dataset_name == 'laion_10k':
            train_dataset_len = 10000
            subset_size = 10000 // self.args.clients_num
            # 将数据集随机分割成十个子集的索引
            indices = list(range(train_dataset_len))
            subsets_indices = [indices[i * subset_size: (i + 1) * subset_size] for i in range(self.args.clients_num)]
            for i in range(self.args.clients_num):
                train_dataset = ObjectAttributeDataset(
                instance_data_root=self.args.instance_data_dir,
                class_prompt=self.args.class_prompt,
                tokenizer=self.tokenizer,
                size=self.args.resolution,
                center_crop=self.args.center_crop,
                random_flip = self.args.random_flip,
                prompt_json = self.args.instance_prompt_loc,
                duplication = self.args.duplication,
                args = self.args
                )
                train_dataset.samples =[train_dataset.samples[j] for j in subsets_indices[i]]
                train_dataset.targets = [train_dataset.targets[j] for j in subsets_indices[i]]
                if self.args.trainsubset is not None:
                    train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, int(len(train_dataset)*self.args.trainsubset))))
                    choicelist = choicelist[:int(len(train_dataset)*self.args.trainsubset)]
            
                if self.args.duplication in ['dup_both','dup_image']:
                    sampler = torch.utils.data.WeightedRandomSampler(train_dataset.samplingweights, len(train_dataset), replacement=True) 
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        shuffle=False,
                        collate_fn=lambda examples: collate_fn(examples),
                        num_workers=self.args.num_workers,
                        sampler = sampler
                    )
                else:
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=self.args.train_batch_size,
                        shuffle=True,
                        collate_fn=lambda examples: collate_fn(examples),
                        num_workers=self.args.num_workers,
                    )
                client = Client(i, train_dataset,
                                    train_dataloader)

                self.clients_set.append(client)
    def local_train_all(self, args, accelerator, OBJECTS, round, c):
        global_parameters = self.clients_set[c].local_train(args, accelerator, OBJECTS, round, c)
        return global_parameters

                