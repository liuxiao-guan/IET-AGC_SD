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
import os
import sys

# 获取当前文件所在目录的上一级目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 将上一级目录路径添加到 sys.path 中
sys.path.append(parent_dir)
from datasets import ObjectAttributeDataset, get_classnames
from utils.draw_utils import concat_h
from clients_acce_es import ClientsGroup
from multiprocessing.pool import ThreadPool as Pool


logger = get_logger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    # parser.add_argument(
    #     "--class_data_dir",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="A folder containing the training data of class images.",
    # )
    parser.add_argument(
        "--instance_prompt_loc",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="nolevel",
        choices = ["nolevel","classlevel","instancelevel_blip","instancelevel_random"],
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    # parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    # parser.add_argument(
    #     "--num_class_images",
    #     type=int,
    #     default=100,
    #     help=(
    #         "Minimal class images for prior preservation loss. If there are not enough images already present in"
    #         " class_data_dir, additional images will be sampled with class_prompt."
    #     ),
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--generation_seed", type=int, default=1024, help="A seed for generation images.")
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save images every X updates steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("-j","--num_workers", type=int, default=0)
    parser.add_argument("--modelsavesteps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--duplication",
        type=str,
        default= "nodup",
        choices = ['nodup','dup_both','dup_image'],
        help="Duplicate the training data or not",
    )

    parser.add_argument(
        "--unet_from_scratch",
        type=str,
        default= "no",
        choices = ['no','yes'],
        help="Duplicate the training data or not",
    )
    parser.add_argument(
        "--weight_pc", type=float, default=0.05, help="Percentage of points to sample more."
    )
    parser.add_argument(
        "--dup_weight", type=int, default=5, help="How likely dup points are, over the others."
    )
    parser.add_argument(
        "--rand_noise_lam", type=float, default=0, help="How much gaussian noise to add to text encoder embedding during training"
    )
    parser.add_argument(
        "--mixup_noise_lam", type=float, default=0, help="How much mixup noise to add to text encoder embedding during training"
    )
    
    parser.add_argument(
        "--trainspecial", type=str, default=None, choices = ['allcaps','randrepl','randwordadd','wordrepeat'],help="which caps to use"
    )
    parser.add_argument(
        "--trainspecial_prob", type=float, default=0.1, help="for special training, intervention probability"
    )
    parser.add_argument(
        "--trainsubset", type=float, default=None, help="percentage of training data to use"
    )
    #### fedderated training
    parser.add_argument(
        "--dataset_name", type=str, default="laion_10k",help="which caps to use"
    )
    parser.add_argument(
        "--clients_num", type=int, default=2, help="the number of data shards"
    )

    parser.add_argument(
        "--total_round", type=int, default=5, help="the round of global training."
    )
    parser.add_argument(
        "--modelsaverounds", type=int, default=5, help="the round of saving model."
    )
    parser.add_argument(
        "--trainggpu", type=int, default=1, help="the number of gpu for training"
    )
    parser.add_argument(
        "--mem_ratio", type=float, default=1.25, help="the skipping threshold"
    )
    parser.add_argument(
        "--arr_ema_decay", type=float, default=0.8, help="the smoothing factor of loss "
    )
    
    
    ################ End of args #####################

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


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


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    OBJECTS = get_classnames(args.instance_data_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=logging_dir
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
         # Handle the repository creation
        args_dict = vars(args)

        # 指定保存文件的文件夹路径
        folder_path = args.output_dir

        # 如果文件夹不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 指定保存文件的路径
        file_path = os.path.join(folder_path, 'args.yaml')
        import yaml 
        ## 将参数保存到文件中
        with open(file_path, 'w') as f:
            yaml.dump(args_dict, f, sort_keys=False)

        print(f"参数已保存到 {file_path} 文件中。")
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import ipdb; ipdb.set_trace()
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load models and create wrapper for stable diffusion
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",
    #     revision=args.revision,
    # )
    if args.unet_from_scratch == 'no':
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )
    else:
        import json
        unet_config = json.loads(open('./unet_config.json').read())
        unet = UNet2DConditionModel(**unet_config)



    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
   
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    clients_group = ClientsGroup(args,tokenizer)
    device_list = []
    for i in range(args.clients_num):
        device_list.append(torch.device(f"cuda:{i}"))
    #device_list = [torch.device("cuda:0"),torch.device("cuda:1")]
    # init local_parameters
    for i in range(0,args.clients_num):
        clients_group.clients_set[i].init(unet,tokenizer,accelerator,args,logger,device_list[i])
    client_idx = [x for x in range(args.clients_num)]
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    import wandb
    if accelerator.is_main_process:
        # accelerator.init_trackers("diffrep_ft", 
        #                           init_kwargs={"wandb":{"name":f"{args.output_dir}_{args.class_prompt}"}, "settings":wandb.Settings(start_method="fork")},
        #                           config=vars(args))
        init_kwargs = {"wandb":{"name":f"{args.output_dir}_{args.class_prompt}","settings":{"console": "off"}}}

        accelerator.init_trackers("diffrep_ft", 
                                init_kwargs=init_kwargs,
                                config=vars(args))
    pool = Pool(processes=args.clients_num)
   # start training
    for round in tqdm(range(1, (args.total_round+1)), desc='Server rounds', unit='round'):
        sum_parameters = None
        train_data_sum = 10000
        # train
        weight_cache = {c: len(clients_group.clients_set[c].train_dataset.targets) / train_data_sum for c in client_idx}
        tasks = [(clients_group,args, accelerator, OBJECTS, round, c) for c in client_idx]
        # 阻塞到任务列表中所有任务完成再往下执行 并且传入多个参数
        results = pool.starmap(train_clients, tasks)

        #results = pool.starmap(partial(clients_group.local_train_all, clients_group=clients_group, round=round), tasks)
        
        
        for global_parameters in results:

            #global_parameters = clients_group.clients_set[c].local_train(args, accelerator, OBJECTS, round, c)
            if sum_parameters is None:
                sum_parameters = {key: var.cpu() for key, var in global_parameters.items()}
            else:
                sum_parameters = {
                    var: sum_parameters[var] + global_parameters[var].cpu()
                    for var in global_parameters
                }
    
        # 遍历模型的参数，并将每个参数的值减半
        for name in sum_parameters:
            sum_parameters[name] = sum_parameters[name]/args.clients_num
        # fedavg
        for c in client_idx:
            clients_group.clients_set[c].set_global_parameters(
                sum_parameters)
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process and round % args.modelsaverounds == 0:
            unet.load_state_dict(sum_parameters)
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unet,
                text_encoder=text_encoder,
                revision=args.revision,
            )
            pipeline.save_pretrained(os.path.join(args.output_dir, f'checkpoint_{round}')) 
    pool.close()
    pool.join()      
    accelerator.end_training()

def train_clients(clients_group,args, accelerator, OBJECTS, round, c):
    return clients_group.local_train_all(args, accelerator, OBJECTS, round, c)  


if __name__ == "__main__":
    args = parse_args()

    assert not (args.duplication == 'dup_image' and args.class_prompt == 'instancelevel_ogcap'), "Duplicating just the image in original captions scenario is not acceptable"

    if args.trainspecial:
        if args.class_prompt != 'instancelevel_blip':
            raise Exception("Cant train special without blip captions")
    
    if args.trainsubset is not None:
         args.output_dir = f"{args.output_dir}_{args.trainsubset}subset"
    if args.unet_from_scratch == 'no':
        args.output_dir = f"{args.output_dir}_{args.class_prompt}_{args.duplication}"
    else:
        args.output_dir = f"{args.output_dir}_{args.class_prompt}_{args.duplication}_unetfromscr"

    if args.duplication in ['dup_both','dup_image']:
        args.output_dir  = f"{args.output_dir}_{args.weight_pc}_{args.dup_weight}"
    
    if args.rand_noise_lam > 0:
        args.output_dir  = f"{args.output_dir}_glam{args.rand_noise_lam}"
    if args.mixup_noise_lam > 0:
        args.output_dir  = f"{args.output_dir}_mixlam{args.mixup_noise_lam}"
    if args.trainspecial is not None:
        args.output_dir  = f"{args.output_dir}_special_{args.trainspecial}_{args.trainspecial_prob}"
    if args.gradient_accumulation_steps != 1:
        args.output_dir = f"{args.output_dir}_gras{args.gradient_accumulation_steps}"
    
    args.output_dir = f"{args.output_dir}_bs{args.train_batch_size}_lr{args.learning_rate}_gpu{args.trainggpu}_mr{args.mem_ratio}_maxst{args.max_train_steps}"


    # TODO: adding noise in text case
    main(args)
