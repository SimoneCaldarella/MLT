import os
import torch
import random

from torch.utils.data import Dataset

class MLTDataset(Dataset):
    """Dataset class for training the
    novelty detection model.
    """
    def __init__(self, configs, preprocessor, loader):
        self.preprocessor = preprocessor
        self.loader = loader
        self.samples = self.loader.load()
        # Add code here
        # Remove following line
        raise NotImplementedError("Write your own init")
        
            
    def __getitem__(self, index):

        # (OPTIONAL)

        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
    
    
    
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