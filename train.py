import os
import random
from typing import Callable, Optional
import numpy as np
import torch
import draccus
import torch.distributed as dist
import argparse
from training.fsdp import FSDPStrategy
from overwatch.overwatch import initialize_overwatch
from mm_utils.utils import *

# nohup bash scripts/pretrain_8_a100.sh > pretrain_8_a100.out 2>&1 &  3269779
# nohup bash scripts/grounded_8_a100.sh > grounded_8_a100.out 2>&1 &  1173215
# nohup bash scripts/phi3.5_grounded_wo_token_4_a40.sh > phi3.5_grounded_wo_token_4_a40.out 2>&1 &

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model', type=str, default='llava_next_video', choices=['llava_next_video'])
    parser.add_argument('--llm', type=str, default='phi3.5', choices=['llama3', 'vicuna', 'phi3.5'])

    parser.add_argument('--dataset', type=str, default='mix_sft', choices=['mix_pretrain', 'mix_grounded', 'mix_sft'])
    parser.add_argument('--max_txt_len', type=int, default=2048)
    parser.add_argument('--num_temporal_tokens', type=int, default=300)
    parser.add_argument('--num_frames', type=int, default=96)
    parser.add_argument('--num_segs', type=int, default=12)
    parser.add_argument('--stage', type=str, default="sft", choices=['pretrain', 'grounded', 'sft'])
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", choices=['eager', 'flash_attention_2']) # choose 'eager' if you cannot install flash_attention_2

    parser.add_argument('--sharding_strategy', type=str, default="full-shard", choices=['shard-grad-op', 'full-shard']) # shard-grad-op for pretrain, full-shard for SFT
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lora_lr', type=float, default=2e-4)
    parser.add_argument('--mm_proj_lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=2e-5) # 1e-3 for pretrain, 2e-5 for SFT
    parser.add_argument('--global_batch_size', type=int, default=128)
    parser.add_argument('--per_device_batch_size', type=int, default=16)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--lr_scheduler_type', type=str, default="linear-warmup+cosine-decay")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default='')
    parser.add_argument('--resume_interval', type=float, default=0.1, help='save resume_interval times per epoch')

    parser.add_argument('--config_path', type=str, default="weight_path/Phi-3.5-vision-instruct")
    parser.add_argument('--tokenizer_path', type=str, default="weight_path/Phi-3.5-mini-instruct")
    parser.add_argument('--pretrained_video_path', type=str, default='weight_path/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt')
    parser.add_argument('--pretrained_vision_proj_llm_path', type=str, default='weight_path/Phi-3.5-vision-instruct-seperated/')
    
    parser.add_argument('--data_dir', type=str, default='/home/haibo/data')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--pretrained_proj', type=str, default='')

    args = parser.parse_args()
    return args

def worker_init_function(worker_id: int) -> None:
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_seq.generate_state(4))
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"
    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return worker_init_function if get_worker_init_fn else None

def pretrain(args) -> None:
    overwatch.info("VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    worker_init_fn = set_global_seed(args.seed, get_worker_init_fn=True)

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating VLM")

    if args.model == 'llava_next_video':
        from models.llava_next_video import LLAVA_NEXT_VIDEO

        model = LLAVA_NEXT_VIDEO(
            dtype=torch.bfloat16, 
            stage=args.stage, 
            max_txt_len=args.max_txt_len, 
            num_frames=args.num_frames,
            num_segs=args.num_segs,
            num_temporal_tokens=args.num_temporal_tokens,
            lora=args.lora,
            llm=args.llm,
            attn_implementation=args.attn_implementation,
            config_path=args.config_path,
            tokenizer_path=args.tokenizer_path,
            pretrained_video_path=args.pretrained_video_path,
            pretrained_vision_proj_llm_path=args.pretrained_vision_proj_llm_path, 
            )
            
        model.vision_tower.to(model.dtype)
        model.video_encoder.to(model.dtype)
        model.multi_modal_projector.to(model.dtype)
        model.video_projecter.to(model.dtype)

        if args.stage in ['grounded', 'sft'] and len(args.pretrained_proj) > 0 and args.llm in args.pretrained_proj:
            ckpt = torch.load(args.pretrained_proj, map_location='cpu')['model']
            if 'multi_modal_projector' in ckpt.keys():
                model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
            if 'video_projecter' in ckpt.keys():
                model.video_projecter.load_state_dict(ckpt['video_projecter'])
            if 'language_model' in ckpt.keys():
                model.language_model.load_state_dict(ckpt['language_model'])

    if overwatch.is_rank_zero():
        print(get_parameter_number(model))

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset")
    if args.dataset == 'mix_pretrain':
        from datasets.mix_pretrain import MixPretrain
        train_dataset = MixPretrain(
        anno_path = os.path.join(args.data_dir, 'mix_pretrain/mix_pretrain.json'),
        video_path = args.data_dir,
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='middle',
        llm=args.llm,
        )
    elif args.dataset == 'mix_grounded':
        from datasets.mix_grounded import MixGrounded
        train_dataset = MixGrounded(
        anno_path = os.path.join(args.data_dir, 'mix_grounded/mix_grounded.json'),
        video_path = args.data_dir,
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='middle',
        llm=args.llm,
        )
    elif args.dataset == 'mix_sft':
        from datasets.mix_sft import MixSFT
        train_dataset = MixSFT(
        anno_path = os.path.join(args.data_dir, 'mix_sft/mix_sft.json'),
        video_path = args.data_dir,
        num_frames = args.num_frames,
        num_segs = args.num_segs,
        num_temporal_tokens = args.num_temporal_tokens,
        sample='middle',
        llm=args.llm,
        )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy")
    train_strategy = FSDPStrategy(
        args=args,
        vlm=model,
        device_id=device_id,
        epochs=args.epoch,
        max_steps=None,
        global_batch_size=args.global_batch_size,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        enable_gradient_checkpointing=True,
        enable_mixed_precision_training=True,
        reduce_in_full_precision=False,
        worker_init_fn=worker_init_fn,
        sharding_strategy=args.sharding_strategy,
    )

    train_strategy.run_setup(n_train_examples=len(train_dataset))

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, seed=args.seed)

    # Save ckpt
    overwatch.info("Save ckpt")
    train_strategy.save_checkpoint(run_dir=args.save_dir, resume=False)

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Initialize Overwatch =>> Wraps `logging.Logger`
    overwatch = initialize_overwatch(__name__)
    args = parse_args()
    if overwatch.is_rank_zero():
        print(args)
    pretrain(args)