"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdata.stateful_dataloader import StatefulDataLoader

import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from overwatch.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)



# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        args,
        vlm,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.args = args
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.language_model.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def plot_records(self, record_list, record_type):
        #1.训练时先新建个列表，然后将loss值调用列表的append方法存入列表中
        #2.例如列表train_recon_loss，Discriminator_loss...，然后将列表名替换train_recon_loss，Discriminator，利用plot即可画出曲线
        #3.最后将画的图保存成图片，imgpath为自定义的图片保存路径。
        # plt.figure(num = 2, figsize=(640,480))
        plt.switch_backend('Agg')
        plt.figure()
        plt.plot(record_list,'b',label=record_type)
        plt.ylabel(record_type)
        plt.xlabel('iteration')
        plt.legend()
        plt.savefig(f"{record_type}.jpg")
        plt.close('all')

    def reduce_metric(self, metric):
        metric_tensor = torch.tensor(metric).to(overwatch.local_rank())
        dist.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
        metric = metric_tensor.item() / overwatch.world_size()
        return metric

    def resume(self, save, file_path=None, dataloader=None, curr_epoch=0):
        if save:
            state_dict = {
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'curr_epoch': curr_epoch,
            }

            for i in range(overwatch.world_size()):
                if overwatch.rank() == i:
                    torch.save(dataloader.state_dict(), f'experiments/dataloader_{overwatch.rank()}.pth')
            dist.barrier()

            for i in range(overwatch.world_size()):
                dataloader_state_dic = torch.load(f'experiments/dataloader_{i}.pth')
                state_dict[f'dataloader_{i}'] = dataloader_state_dic
            dist.barrier()

            for i in range(overwatch.world_size()):
                if overwatch.rank() == i:
                    torch.save(self.optimizer.state_dict(), f'experiments/optimizer_{overwatch.rank()}.pth')
            dist.barrier()

            for i in range(overwatch.world_size()):
                optimizer_state_dic = torch.load(f'experiments/optimizer_{i}.pth')
                state_dict[f'optimizer_{i}'] = optimizer_state_dic
            dist.barrier()

            if overwatch.is_rank_zero():
                pth_files = glob.glob(os.path.join('experiments/', "*.pth"))
                for pth_file in pth_files:
                    os.remove(pth_file)

            if overwatch.is_rank_zero():
                torch.save(state_dict, file_path)
            self.save_checkpoint(run_dir=self.args.save_dir, resume=True)

        else:
            state_dict = torch.load(file_path)
            curr_epoch = state_dict['curr_epoch']
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

            dist.barrier()
            for i in range(overwatch.world_size()):
                if overwatch.rank() == i:
                    dataloader.load_state_dict(state_dict[f'dataloader_{i}'])
            dist.barrier()
            for i in range(overwatch.world_size()):
                if overwatch.rank() == i:
                    self.optimizer.load_state_dict(state_dict[f'optimizer_{i}'])
            dist.barrier()

            return dataloader, curr_epoch
            

    def run_training(
        self,
        dataset: Dataset,
        seed: int = 42,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        sampler = DistributedSampler(
            dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            num_workers=overwatch.rank(),
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
        )

        curr_epoch = 0
        gone_steps = 0
        if not self.args.not_two_stream:
            resume_state_path = self.args.save_dir + f'/{self.args.stage}_{self.args.model}_{self.args.llm}_{self.args.dataset}_state_dict_resume.pth'
        else:
            resume_state_path = self.args.save_dir + f'/{self.args.stage}_{self.args.model}_{self.args.llm}_{self.args.dataset}_state_dict_resume_not_two_stream.pth'            

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps

        if self.args.resume:
            dataloader, curr_epoch = self.resume(
                save=False, 
                file_path=resume_state_path,
                dataloader=dataloader, 
                )

            if '_snapshot' in dataloader.state_dict().keys():
                gone_steps = dataloader.state_dict()['_snapshot']['_snapshot_step'] // self.grad_accumulation_steps + curr_epoch * steps_per_epoch
            else:
                gone_steps = dataloader.state_dict()['_sampler_iter_yielded'] // self.grad_accumulation_steps + curr_epoch * steps_per_epoch

        dist.barrier()

        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        iteration_loss_list = []
        # iteration_lr_list = [[] for i in range(len(self.optimizer.param_groups))]

        # === Train ===
        # status = metrics.get_status()


        with tqdm(
            total=(
                (self.epochs * steps_per_epoch - gone_steps)
                if self.max_steps is None
                else self.max_steps
            ),
            # desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(curr_epoch, self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                iteration_loss = 0


                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!

                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        samples = {
                            "video_ids": batch['video_ids'],
                            "text_inputs": batch['text_inputs'],
                            "temporal_pixel_values":batch['temporal_pixel_values'],
                            "spatial_pixel_values": batch['spatial_pixel_values'],
                            }


                        outputs = self.vlm(samples)
                        loss = outputs['loss']



                        if torch.isnan(loss).any():
                            print(loss, batch['text_inputs'], batch['video_ids'], batch['question_ids'])
                            raise ValueError('loss Nan !!!!!!!!!!!!!!!')

                    # metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    iteration_loss += normalized_loss.item()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        # metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        iteration_loss_list.append(self.reduce_metric(iteration_loss))
                        iteration_loss = 0
                        # for i in range(len(iteration_lr_list)):
                        #     iteration_lr_list[i].append(self.optimizer.param_groups[i]['lr'])

                        # Push Metrics
                        # metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        # status = metrics.push()

                        # # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        # if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        #     self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                        #     dist.barrier()

                        #     return

                        # Update Progress Bar
                        progress.update()
                        # progress.set_description(status)

                    plot_interval = int(steps_per_epoch*self.grad_accumulation_steps/100)
                    if (train_idx + 1) % plot_interval == 0:
                        if overwatch.is_rank_zero():
                            self.plot_records(iteration_loss_list, f'{self.args.model}_{self.args.llm}_{self.args.stage}_{self.args.dataset}_loss')
                            # for i in range(len(iteration_lr_list)):
                            #     self.plot_records(iteration_lr_list[i], f'{self.args.model}_{self.args.stage}_{self.args.dataset}_{self.trainable_module_keys[i]}_lr')


                    # resume_interval = int(steps_per_epoch*self.grad_accumulation_steps/self.args.resume_interval) # save resume_interval times per epoch
                    # if (train_idx + 1) % resume_interval == 0:
                    #     if overwatch.is_rank_zero():
                    #         print('save an checkpoint for resume')
                    #     self.resume(
                    #         save=True, 
                    #         file_path=resume_state_path,
                    #         dataloader=dataloader, curr_epoch=epoch
                    #     )

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                # self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()