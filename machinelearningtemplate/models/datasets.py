import os
import torch
import random

import _pickle as pkl

from torch.utils.data import Dataset
from utils.misc       import load_config

class FeaturesDS(Dataset):
    """Dataset class for training the
    novelty detection model.
    """
    def __init__(self, id_splits, ood_splits, pre_processor, loader, **kwargs):
        self.pre_processor = pre_processor
        self.loader = loader
        self.samples = []
        self.add_splits_pairs_from_names(id_splits, ood_splits)
        
    def add_splits_pairs_from_names(self, id_splits, ood_splits):
        
        id_splits, ood_splits = self.loader.load(id_splits + ood_splits)
        
        for id_split, ood_split in zip(id_splits, ood_splits):
            features_splits = self.pre_processor.process_separately(id_split, ood_split)
            self.samples += features_splits
            
    def add_splits_pairs_from_features(self, id_splits, ood_splits):
        for id_split, ood_split in zip(id_splits, ood_splits):
            features_splits = self.pre_processor.process_separately(id_split, ood_split)
            self.samples += features_splits
        
    def add_splits(self, splits):
        for split in splits:
            features_splits = self.pre_processor.process_jointly(split)
            self.samples += features_splits
            
    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    
class PreProcessor:
    """Pre-Process features with different operations
    like subsampling or filtering samples.
    """
    
    def __init__(self, 
                 replay_percentage = None,
                 balance_splits = False, 
                 nd_threshold = False, 
                 keep_holdout = False):
        
        self.replay_percentage = replay_percentage
        self.balance_splits = balance_splits
        self.nd_threshold = nd_threshold
        self.keep_holdout = keep_holdout
        
    def process_separately(self, id_split, ood_split) -> list[torch.Tensor, torch.Tensor]:
        
        id_split = self.add_label(id_split, label=1)
        ood_split = self.add_label(ood_split, label=0)
        
        if self.balance_splits:
            id_split, ood_split = self.subsample_larger(id_split, ood_split)
            
        if self.replay_percentage:
            id_split = self.subsample(id_split)
            ood_split = self.subsample(ood_split)
            
        return self.compose_splits(id_split, ood_split)
            
    def add_label(self, split, label) -> list[torch.Tensor, torch.Tensor]:
        
        processed_split = []
        
        for sample in split:
            processed_split.append([torch.Tensor(sample), torch.Tensor([label])])
            
        return processed_split
                
    def subsample(self, split):
        
        split = random.sample(split, int(self.replay_percentage*len(split)))
        
        return split
                
    def subsample_larger(self, id_split, ood_split) -> list[torch.Tensor, torch.Tensor]:
        
        k_samples = min(len(id_split), len(ood_split))
        
        processed_id_split = random.sample(id_split, k_samples)
        processed_ood_split = random.sample(ood_split, k_samples)
        
        return processed_id_split, processed_ood_split
    
    def compose_splits(self, id_split, ood_split) -> list[torch.Tensor, torch.Tensor]:
        
        return id_split + ood_split
    
    
class FeaturesLoader:
    """This component read the computed features file
    and loads the intermediate features.
    """
    def __init__(self):
        self.features_path = load_config("FEATURES_FOLDER")
        self.dataset_name = load_config("TEST_DOMAIN_DATASET")
        self.objd_threshold = load_config(f"{self.dataset_name}_OBJD_THRESHOLD")
        
    def read_file(self, path):
        with open(path, "rb") as file:
            dataset_file = pkl.load(file)
            
        return dataset_file
    
    def extract_obj_features(self, dataset):
        
        filtered_features = []
        
        for sample in dataset:
            if sample["score"] > self.objd_threshold:
                filtered_features.append(sample["inter_feats"])
                
        return filtered_features
        
    def load(self, dataset_names):
        
        id_splits = []
        ood_splits = []
        
        for dataset_name in dataset_names:
            
            dataset = self.read_file(os.path.join(self.features_path, dataset_name))
            dataset = self.extract_obj_features(dataset)
            
            if "overlapped" in dataset_name:
                id_splits.append(dataset)
                
            elif "rm_overlap" in dataset_name:
                ood_splits.append(dataset)
                
        return id_splits, ood_splits