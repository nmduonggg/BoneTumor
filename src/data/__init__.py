"""create dataset and dataloader"""
import torch
import torch.utils.data

def create_dataloader(dataset, dataset_opt, opt=None, shuffle=False, sampler=None, collate_fn=None):
    phase = dataset_opt['phase']
    batch_size = dataset_opt['batch_size']
    num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
    
    if phase == 'train':
        if dataset_opt['balance']:
            weights = make_weights_for_balanced_classes(dataset, dataset.n_classes)
            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))  
            shuffle=False 
        
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                           sampler=sampler, collate_fn=collate_fn, prefetch_factor=2,
                                           persistent_workers=True, drop_last=False, pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                           pin_memory=True, prefetch_factor=2, persistent_workers=True, collate_fn=collate_fn)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'base':
        from data.BaseDataset import BaseDataset as D
    elif mode == 'segment':
        from data.SegmentDataset import SegmentDataset as D
    elif mode == 'classification':
        from data.ClassificationDataset import ClassificationDataset as D
    elif mode == 'journal_classification':
        from data.journal.OneMag_ClassificationDataset import OneMag_ClassificationDataset as D
    elif mode == 'reorder':
        from data.ReorderDataset import ReorderDataset as D
    elif mode == 'classification_multi_mag':
        from data.ClassificationDataset_MultiMag import ClassificationDataset_MultiMag as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    return dataset

def make_weights_for_balanced_classes(dataset, nclasses):
    n_samples = len(dataset)
    count_per_class = [0 for _ in range(nclasses)]
    for _, cid in dataset:
        count_per_class[cid.item()] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_samples) / float(count_per_class[i])
    weights = [0 for _ in range(n_samples)]
    for idx, (x, cid) in enumerate(dataset):
        weights[idx] = weight_per_class[cid.item()]
    return weights