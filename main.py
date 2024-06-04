from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
import numpy as np
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import gdown
import os
from dataset import PushTImageDataset, normalize_data, unnormalize_data
from network import ConditionalUnet1D, replace_bn_with_gn, get_resnet, MultiImageObsEncoder
import torchvision
import wandb
from pytorch_utils import dict_apply
import sys
from typing import Optional
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def create_networks(config: Dict[str, int], device: torch.device) -> nn.ModuleDict:
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = 512 
    lowdim_obs_dim = 4
    obs_dim = vision_feature_dim + lowdim_obs_dim

    noise_pred_net = ConditionalUnet1D(
        input_dim=config['action_dim'],
        global_cond_dim=obs_dim*config['obs_horizon']
    )

    obs_encoder = DDP(vision_encoder.to(device))
    noise_pred_net = DDP(noise_pred_net.to(device))

    nets = nn.ModuleDict({
        'vision_encoder': obs_encoder,
        'noise_pred_net': noise_pred_net
    })
    return nets

def create_noise_scheduler(num_diffusion_iters: int) -> DDPMScheduler:
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    return noise_scheduler

def train(nets: nn.ModuleDict, dataloader: torch.utils.data.DataLoader, device: torch.device, noise_scheduler: DDPMScheduler, dataset_stats: Dict[str, float], config: Dict[str, int], local_rank: int, checkpoint_path: Optional[str] = None) -> nn.ModuleDict:
    wandb.login(key='1bafb483fc18be5411bc07ae3693994f943313c7')
    wandb.init(project="diffusion_test")
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(dataloader) * config['num_epochs'])

    with tqdm(range(config['num_epochs']), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nimage = nbatch['image'][:,:config['obs_horizon']].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:config['obs_horizon']].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                    #assert image_features.shape == (B, config['obs_horizon'], 512*len(nets['vision_encoder'].image_keys))
                    image_features = image_features.reshape(B, config['obs_horizon'], -1)
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    noise = torch.randn(naction.shape, device=device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(nets.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            avg_loss = np.mean(epoch_loss)
            wandb.log({
                "epoch": epoch_idx,
                "loss": avg_loss,
            })
            # Checkpointing EMA weights
            if dist.get_rank() == 0 and epoch_idx % 10 == 0:  # Checkpoint every epoch; adjust as needed
                dist.barrier()
                checkpoint_path = f"checkpoints/ema_checkpoint_epoch_{epoch_idx}.pt"
                ema_state = {
                    'model_state_dict': nets.state_dict(),
                    'epoch': epoch_idx,
                    'ema_state_dict': ema.state_dict(),  # Assuming EMAModel has a state_dict method
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'mse': loss_cpu,
                    'normalization_stats': dataset_stats  # Add normalization stats to the checkpoint
                }
                torch.save(ema_state, checkpoint_path)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    ema_nets = nn.ModuleDict(nets)
    ema.copy_to(ema_nets.parameters())
    return ema_nets

def load_pretrained(nets: nn.ModuleDict, ckpt_path: str) -> Tuple[nn.ModuleDict, Dict[str, float]]:
    if not os.path.isfile(ckpt_path):
        id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
        gdown.download(id=id, output=ckpt_path, quiet=False)

    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_nets = nn.ModuleDict(nets)
    ema_nets.load_state_dict(state_dict['model_state_dict'])
    normalization_stats = state_dict['normalization_stats']
    print('Pretrained weights and normalization stats loaded.')
    return ema_nets, normalization_stats

def main():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    config = {
        'pred_horizon': 16,
        'obs_horizon': 2,
        'action_horizon': 8,
        'num_diffusion_iters': 100,
        'batch_size': 64,
        'num_workers': 4,
        'num_epochs': 100,
        'action_dim': 4
    }
    device = torch.device('cuda')

    dataset_path = "data/output.zarr"

    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=config['pred_horizon'],
        obs_horizon=config['obs_horizon'],
        action_horizon=config['action_horizon']
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    nets = create_networks(config, device=device)

    noise_scheduler = create_noise_scheduler(config['num_diffusion_iters'])

    ema_nets = train(nets, dataloader, device, noise_scheduler, dataset.stats, config, checkpoint_path, local_rank)

if __name__ == '__main__':
    main()
