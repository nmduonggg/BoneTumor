import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import socket

import options.options as option
from data import create_dataloader, create_dataset
from model import create_model
from loss import FocalLoss
import utils.utils as utils
import data.utils as data_utils
from huggingface_hub import login

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port
        return s.getsockname()[1]  # Get assigned port

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'  # Use dynamically assigned free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup_distributed():
    dist.destroy_process_group()

def train(rank, world_size, opt):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:  # Only rank 0 logs in to HF Hub
        login(opt['token'])

    working_dir = os.path.join('.', opt['job_dir'], opt['name'])
    os.makedirs(working_dir, exist_ok=True)
    
    model = create_model(opt).to(device)
    if opt['path']['pretrain_model'] is not None:
        state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        if rank == 0:
            print("[INFO] Loaded pre-trained model")
    
    train_opt = opt['train']
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = utils.create_optimizer(model.parameters(), train_opt)
    
    weight = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2]).to(device)
    weight /= torch.sum(weight)
    loss_func = nn.CrossEntropyLoss(weight=weight)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_opt['epochs'], train_opt['eta_min'])
    
    train_set = create_dataset(opt['datasets']['train'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = create_dataloader(train_set, opt['datasets']['train'], opt, train_sampler)
    
    valid_set = create_dataset(opt['datasets']['valid'])
    valid_loader = create_dataloader(valid_set, opt['datasets']['valid'], opt, None)
    
    if rank == 0 and opt['wandb']:
        wandb.init(project="BoneTumor-Journal", name=opt['name'], group=opt['network_G']['which_model_G'])
    
    best_prec = 0.0
    for epoch in range(train_opt['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        loss_tracker = utils.MetricTracker('Train Loss')
        acc_tracker = utils.MetricTracker('Train Accuracy')
        
        for batch in tqdm(train_loader, total=len(train_loader) // world_size, disable=(rank != 0)):
            im, gt = batch
            im, gt = im.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(im)
            loss = loss_func(pred, gt)
            loss.backward()
            optimizer.step()
            
            loss_tracker.update(loss.item(), im.size(0))
            acc_tracker.update((pred.argmax(dim=1) == gt).float().mean().item(), im.size(0))
        
        lr_scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss_tracker} | Accuracy: {acc_tracker}")
            
            eval_loss, eval_acc = evaluate(model, valid_loader, loss_func, device)
            if eval_acc > best_prec:
                best_prec = eval_acc
                torch.save(model.module.state_dict(), os.path.join(working_dir, '_best.pt'))
            
            if opt['wandb']:
                wandb.log({"train_loss": loss_tracker.avg, "train_acc": acc_tracker.avg, "val_loss": eval_loss, "val_acc": eval_acc})
    
    cleanup_distributed()

def evaluate(model, valid_loader, loss_func, device):
    model.eval()
    loss_tracker = utils.MetricTracker('Valid Loss')
    acc_tracker = utils.MetricTracker('Valid Accuracy')
    
    with torch.no_grad():
        for im, gt in tqdm(valid_loader, total=len(valid_loader), disable=True):
            im, gt = im.to(device), gt.to(device)
            pred = model(im)
            loss = loss_func(pred, gt)
            loss_tracker.update(loss.item(), im.size(0))
            acc_tracker.update((pred.argmax(dim=1) == gt).float().mean().item(), im.size(0))
    
    return loss_tracker.avg, acc_tracker.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-root', type=str, default=None, choices=['.'])
    args = parser.parse_args()
    opt = option.parse(args.opt, root=args.root)
    opt = option.dict_to_nonedict(opt)
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} devices")
    mp.spawn(train, args=(world_size, opt), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
