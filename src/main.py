import os
import datetime
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

import options.options as option
from data import create_dataloader, create_dataset
from model import create_model
import utils.utils as utils




abspath = os.path.abspath(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

opt = option.dict_to_nonedict(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % opt['gpu_ids'][0]
device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

for phase, dataset_opt in opt['datasets'].items():
    if phase=='train': 
        train_set = create_dataset(dataset_opt)
        train_loader = create_dataloader(train_set, dataset_opt, opt, None)
    elif phase=='valid': 
        valid_set = create_dataset(dataset_opt)
        valid_loader = create_dataloader(valid_set, dataset_opt, opt, None)
    elif phase=='test': 
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt, None)
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    
working_dir = os.path.join('.', opt['job_dir'], opt['name'])
os.makedirs(working_dir, exist_ok=True)
    
model = create_model(opt)

# fix and load weight
if opt['path']['pretrain_model'] is not None:
    state_dict = torch.load(opt['path']['pretrain_model'], map_location='cpu')
    current_dict = model.state_dict()
    new_state_dict={k:v if v.size()==current_dict[k].size()  else  current_dict[k] for k,v in zip(current_dict.keys(), state_dict.values())}    # fix the size of checkpoint state dict
    _strict=True
    if opt['name'] == 'ProvGigaPath':   # Not load trained weight but the quantized pretrained weight for FM encoder
        new_state_dict = {k: v for k, v in new_state_dict.items() if 'classifier' in k}
        _strict=False
    
    model.load_state_dict(new_state_dict, strict=_strict)  
    print("[INFO] Load weight from:", opt['path']['pretrain_model'])

train_opt = opt['train']
optimizer = utils.create_optimizer(model.parameters(), train_opt)

weight = torch.tensor([1.0, 1.0, 2.0])
weight /= weight.sum()
loss_func = nn.CrossEntropyLoss(weight=weight.to(device))

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_opt['epochs'], train_opt['eta_min'])

def train():
    
    model.to(device)
    
    #### Initialization ####
    loss_tracker = utils.MetricTracker('Train Loss')
    acc_tracker = utils.MetricTracker('Train Accuracy')
    best_acc = 0.0
    #### Start Training ####
    for epoch in range(train_opt['epochs']):
        
        if epoch % train_opt['val_freq']==0:    # Validation
            eval_loss, eval_acc = evaluate()
            print(f"[TEST] Epoch {epoch}|{eval_loss}|{eval_acc}")
            
            if eval_acc.avg > best_acc:
                print(f"[WARN] Save best performance model at epoch {epoch}!")
                best_acc = eval_acc.avg
                torch.save(model.state_dict(), os.path.join(working_dir, '_best.pt'))
            
        model.train()
        
        for im, gt in tqdm(train_loader, total=len(train_loader)):
            batch_size = im.shape[0]
            im = im.to(device)
            gt = gt.to(device)
            
            pred = model(im)
            loss = loss_func(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_tracker.update(loss.detach().cpu().item(), batch_size)
            acc_tracker.update(utils.compute_acc(pred, gt), batch_size)
            
        print(f"[Train] Epoch {epoch}|{loss_tracker}|{acc_tracker}")
        
        lr_scheduler.step()
        
        loss_tracker.reset()
        acc_tracker.reset()
            
    return

def evaluate():
    
    loss_tracker = utils.MetricTracker('Valid Loss')
    acc_tracker = utils.MetricTracker('Valid Accuracy')
    model.to(device)
    model.eval()
    
    for im, gt in tqdm(valid_loader, total=len(valid_loader)):
        batch_size = im.shape[0]
        im = im.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            pred = model(im)
            
        loss = loss_func(pred, gt)
        loss_tracker.update(loss.detach().cpu().item(), batch_size)
        acc_tracker.update(utils.compute_acc(pred, gt), batch_size)
        
    return loss_tracker, acc_tracker

def test():
    
    loss_tracker = utils.MetricTracker('Test Loss')
    acc_tracker = utils.MetricTracker('Test Accuracy')
    model.to(device)
    model.eval()
    labels, preds = list(), list()
    for im, gt in tqdm(test_loader, total=len(test_loader)):
        batch_size = im.shape[0]
        im = im.to(device)
        gt = gt.to(device)
        
        with torch.no_grad():
            pred = model(im)
            
        loss = loss_func(pred, gt)
        loss_tracker.update(loss.detach().cpu().item(), batch_size)
        acc_tracker.update(utils.compute_acc(pred, gt), batch_size)
        
        labels.extend(gt.cpu().tolist())
        preds.extend(torch.argmax(pred.cpu(), dim=1).tolist())
        
    utils.compute_all_metrics(preds, labels)
    print(f"{loss_tracker}|{acc_tracker}")
    
    return loss_tracker, acc_tracker
        
    
if __name__ == '__main__':
    if opt['is_train']:
        print("[INFO] Start training...")
        train()
    elif opt['is_test']:
        print("[INFO] Start testing...")
        test()
        
        
        
        
        