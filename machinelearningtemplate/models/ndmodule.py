import torch

from tqdm            import tqdm
from torch.nn        import functional as F
from utils.misc      import load_config
from sklearn.cluster import KMeans
from modeling.supcon import (SupConLoss, 
                             warmup_learning_rate, 
                             adjust_learning_rate)

# -----------------------------------------------------------------

class ND(torch.nn.Module):
    """Class that act as interface
    for all ND models
    """
    def __init__(self, input_size=2048):
        super().__init__()
        
        self.device = load_config("DEVICE")
        
        self.input_size = input_size
        self.num_layers = 5
        
        self.layer1 = torch.nn.Linear(input_size, 1024)
        self.layer1_norm = torch.nn.LayerNorm([1024])
        
        self.layer2 = torch.nn.Linear(1024, 512)
        self.layer2_norm = torch.nn.LayerNorm([512])
        
        self.layer3 = torch.nn.Linear(512, 256)
        self.layer3_norm = torch.nn.LayerNorm([256])
        
        self.layer4 = torch.nn.Linear(256, 128)
        self.layer4_norm = torch.nn.LayerNorm([128])
        
        self.output = torch.nn.Linear(128, 1)
        
        self.dropout = torch.nn.Dropout(load_config("DROPOUT_P"))
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        raise NotImplementedError
    
    def train_epoch(self, train_dl, optimizer):
        raise NotImplementedError
    
    def compute_loss(self, features, labels):
        raise NotImplementedError
    
    def compute_score(self, x):
        raise NotImplementedError


# -------------------------------------------------------------------------------


class NDBCE(ND):
    """ND model with BCE.
    """
    
    def __init__(self, input_size=2048):
        
        super().__init__(input_size)
        self.to(self.device)
        
    def forward(self, x):
        for i in range(1, self.num_layers):
            
            x = getattr(self, f"layer{i}")(x)
            
            if load_config("LAYER_NORM") == "on":
                x = getattr(self, f"layer{i}_norm")(x)
            x = self.relu(x)
            
            if load_config("DROPOUT") == "on":
                x = self.dropout(x)
        
        output = self.output(x)
        return output
    
    
    def compute_loss(self, features, labels):
        
        output = self.forward(features)
        loss = F.binary_cross_entropy_with_logits(output.squeeze(), labels.squeeze())
        return loss
        
    def compute_score(self, x):
        
        self.eval()
        return self.forward(x)
    
    def train_epoch(self, train_dl, optimizer):
        
        epoch_loss = []
        self.train()
        
        for s in tqdm(train_dl, desc="Training"):
            # s = [intermediate_features: Tensor, labels: Tensor]
            # s.shape = [batch_size x interfeats_size, batch_size]
            s = [f.to(self.device) for f in s]
            
            optimizer.zero_grad()
            loss = self.compute_loss(*s)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            
        return torch.mean(torch.stack(epoch_loss))
    
    def valid_epoch(self, dataloader):
        
        epoch_loss = []
        self.eval()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Validation"):
                # s = [intermediate_features: Tensor, labels: Tensor]
                # s.shape = [batch_size x interfeats_size, batch_size]
                s = [f.to(self.device) for f in s]
                loss = self.compute_loss(*s)
                
                epoch_loss.append(loss)
            
        return torch.mean(torch.stack(epoch_loss))
        
    
    def set_best_weights(self, stopper):
        
        if stopper.is_active:
            self.load_state_dict(stopper.best_model_state_dict)
    
    def run_training(self,
                     dataloaders,
                     epochs,
                     optimizer, 
                     scheduler, 
                     stopper, 
                     logger):
        
        scheduler_name = load_config("SCHEDULER_NAME")
        
        for e in range(epochs):
            logger.info(f"Start Epoch {e}")
            
            train_loss = self.train_epoch(dataloaders["train"], optimizer)
            valid_loss = self.valid_epoch(dataloaders["valid"])
            
            if scheduler_name == "ReduceLROnPlateau":
                scheduler.step(valid_loss)
            else:
                scheduler.step()
            
            end_train = stopper.step(valid_loss, self.state_dict())
            
            train_loss = train_loss.detach().cpu().numpy()
            valid_loss = valid_loss.detach().cpu().numpy()
            
            logger.info(f"Train Loss: {train_loss} | Valid Loss: {valid_loss}")
            logger.add_values({"train_loss": train_loss, "valid_loss": valid_loss})
            
            if end_train:
                logger.info("Training Ended")
                break
            
        self.set_best_weights(stopper)
        
    def generate_predictions(self, dataloader):
        
        scores = []
        labels = []
        
        self.eval()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Generating Predictions"):
                s = [f.to(self.device) for f in s]
                score = self.compute_score(s[0])
                scores += score.squeeze().tolist()
                labels += s[1].squeeze().tolist()
                
        return scores, labels
            
                    
        
class NDSUPCON(ND):
    
    def __init__(self, input_size=2048):
        
        super().__init__(input_size)
        self.supcon = SupConLoss()
        self.to(self.device)
        
        for i in range(1, self.num_layers):
            torch.nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)
            
    def forward(self, x):
        for i in range(1, self.num_layers-1):
            
            x = getattr(self, f"layer{i}")(x)
            
            if load_config("LAYER_NORM") == "on":
                x = getattr(self, f"layer{i}_norm")(x)
            x = self.relu(x)
            
            if load_config("DROPOUT") == "on":
                x = self.dropout(x)
                
        output = self.layer4(x)
        
        return output
    
    
    def compute_loss(self, features, labels):
        
        output = self.forward(features)
        output = F.normalize(output, dim=-1)
        
        # Unsqueeze in the second dimension in order to 
        # match the right size requested by SupConLoss. 
        # However I have only one view/transformation per sample
        
        loss = self.supcon(output.squeeze().unsqueeze(1), 
                           labels.squeeze().long())
        return loss
        
    def compute_score(self, x):
        
        self.eval()
        
        x = self.forward(x)
        
        distances = torch.matmul(F.normalize(x.to(self.device), dim=-1), 
                                 F.normalize(self.k_id_centroids.to(self.device), dim=-1).T)
        score = distances.sort(descending=True).values[:, 0].cpu()
        
        return score
    
    def train_epoch(self, train_dl, optimizer, epoch):
        
        epoch_loss = []
        self.train()
        
        for i, s in enumerate(tqdm(train_dl, desc="Training")):
            # s = [intermediate_features: Tensor, labels: Tensor]
            # s.shape = [batch_size x interfeats_size, batch_size]
            s = [f.to(self.device) for f in s]
            
            optimizer.zero_grad()
            loss = self.compute_loss(*s)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            
            warmup_learning_rate(epoch, load_config("LR"), i, len(train_dl), optimizer, load_config("MAXIMUM_EPOCHS"))
            
        return torch.mean(torch.stack(epoch_loss))
    
    def valid_epoch(self, dataloader):
        
        epoch_loss = []
        self.eval()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Validation"):
                # s = [intermediate_features: Tensor, labels: Tensor]
                # s.shape = [batch_size x interfeats_size, batch_size]
                s = [f.to(self.device) for f in s]
                loss = self.compute_loss(*s)
                
                epoch_loss.append(loss)
            
        return torch.mean(torch.stack(epoch_loss))
        
    
    def set_best_weights(self, stopper):
        
        if stopper.is_active:
            self.load_state_dict(stopper.best_model_state_dict)
    
    def run_training(self,
                     dataloaders,
                     epochs,
                     optimizer, 
                     scheduler, 
                     stopper, 
                     logger):
        
        self.train_dl = dataloaders["train"]
        
        for e in range(epochs):
            logger.info(f"Start Epoch {e}")
            train_loss = self.train_epoch(dataloaders["train"], optimizer, e)
            valid_loss = self.valid_epoch(dataloaders["valid"])
            
            adjust_learning_rate(optimizer, load_config("LR"), e, load_config("MAXIMUM_EPOCHS"))
            
            end_train = stopper.step(valid_loss, self.state_dict())
            
            train_loss = train_loss.detach().cpu().numpy()
            valid_loss = valid_loss.detach().cpu().numpy()
            
            logger.info(f"Train Loss: {train_loss} | Valid Loss: {valid_loss}")
            logger.add_values({"train_loss": train_loss, "valid_loss": valid_loss})
            
            if end_train:
                logger.info("Training Ended")
                break
            
        self.set_best_weights(stopper)
        
    def generate_predictions(self, dataloader):
        
        scores = []
        labels = []
        
        self.eval()
        
        self.prepare_scoring()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Generating Predictions"):
                s = [f.to(self.device) for f in s]
                score = self.compute_score(s[0])
                scores += score.squeeze().tolist()
                labels += s[1].squeeze().tolist()
                
        return scores, labels
    
    def prepare_scoring(self):
    
        id_features = torch.Tensor().to(self.device)
        
        for s in self.train_dl:
            s = [f.to(self.device) for f in s]
            ot = self.forward(s[0])
            
            id_features = torch.cat([id_features, ot[(s[1].squeeze()).bool()]])
            
        kmeans_id = KMeans(n_clusters=load_config("KMEANS_NUM"), n_init=10, random_state=0).fit(id_features.cpu().detach().numpy())
        self.k_id_centroids = torch.Tensor(kmeans_id.cluster_centers_)
    

class ICONP(ND):
    
    def __init__(self, input_size=2048):
        
        super().__init__(input_size)
        self.supcon = SupConLoss(temperature=0.1)
        self.to(self.device)
        
        for i in range(1, self.num_layers):
            torch.nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)
            
    def supcon_forward(self, x):
        for i in range(1, self.num_layers-1):
            
            x = getattr(self, f"layer{i}")(x)
            
            if load_config("LAYER_NORM") == "on":
                x = getattr(self, f"layer{i}_norm")(x)
            x = self.relu(x)
            
            if load_config("DROPOUT") == "on":
                x = self.dropout(x)
                
        output = self.layer4(x)
        
        return output
            
    def forward(self, x):
        for i in range(1, self.num_layers-1):
            
            x = getattr(self, f"layer{i}")(x)
            
            if load_config("LAYER_NORM") == "on":
                x = getattr(self, f"layer{i}_norm")(x)
            
            x = self.relu(x)
            
            if load_config("DROPOUT") == "on":
                x = self.dropout(x)
                
        x = self.layer4(x)
        
        output = self.output(x)
        
        return output
    
    
    def compute_loss(self, features, labels):
        
        gamma = load_config("GAMMA")
        
        output = self.supcon_forward(features)
        supcon_output = F.normalize(output, dim=-1)
        
        detached = load_config("DETACHED")
        
        if detached == "on":
            bce_output = self.output(supcon_output.detach().clone())
        
        else:
            bce_output = self.output(supcon_output)
        
        # Unsqueeze in the second dimension in order to 
        # match the right size requested by SupConLoss. 
        # However I have only one view/transformation per sample
        
        supcon = self.supcon(supcon_output.squeeze().unsqueeze(1), 
                             labels.squeeze().long())
        
        bce = F.binary_cross_entropy_with_logits(bce_output.squeeze(), 
                                                 labels.squeeze())
        loss = supcon + gamma*bce
        
        return loss
        
    def compute_score(self, x):
        
        self.eval()
        
        score = self.forward(x)
        
        return score
    
    def train_epoch(self, train_dl, optimizer, epoch):
        
        epoch_loss = []
        self.train()
        
        for i, s in enumerate(tqdm(train_dl, desc="Training")):
            # s = [intermediate_features: Tensor, labels: Tensor]
            # s.shape = [batch_size x interfeats_size, batch_size]
            s = [f.to(self.device) for f in s]
            
            optimizer.zero_grad()
            loss = self.compute_loss(*s)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            
            warmup_learning_rate(epoch, load_config("LR"), i, len(train_dl), optimizer, load_config("MAXIMUM_EPOCHS"))
            
        return torch.mean(torch.stack(epoch_loss))
    
    def valid_epoch(self, dataloader):
        
        epoch_loss = []
        self.eval()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Validation"):
                # s = [intermediate_features: Tensor, labels: Tensor]
                # s.shape = [batch_size x interfeats_size, batch_size]
                s = [f.to(self.device) for f in s]
                loss = self.compute_loss(*s)
                
                epoch_loss.append(loss)
            
        return torch.mean(torch.stack(epoch_loss))
        
    
    def set_best_weights(self, stopper):
        
        if stopper.is_active:
            self.load_state_dict(stopper.best_model_state_dict)
    
    def run_training(self,
                     dataloaders,
                     epochs,
                     optimizer, 
                     scheduler, 
                     stopper, 
                     logger):
        
        self.train_dl = dataloaders["train"]
        
        for e in range(epochs):
            logger.info(f"Start Epoch {e}")
            train_loss = self.train_epoch(dataloaders["train"], optimizer, e)
            
            if dataloaders["valid"]:
                valid_loss = self.valid_epoch(dataloaders["valid"])
                
                adjust_learning_rate(optimizer, load_config("LR"), e, load_config("MAXIMUM_EPOCHS"))
                
                end_train = stopper.step(valid_loss, self.state_dict())
                
                train_loss = train_loss.detach().cpu().numpy()
                valid_loss = valid_loss.detach().cpu().numpy()
                
                logger.info(f"Train Loss: {train_loss} | Valid Loss: {valid_loss}")
                logger.add_values({"train_loss": train_loss, "valid_loss": valid_loss})
                
                if end_train:
                    logger.info("Training Ended")
                    break
        
        if dataloaders["valid"]:
            self.set_best_weights(stopper)
        
    def generate_predictions(self, dataloader):
        
        scores = []
        labels = []
        
        self.eval()
        
        with torch.no_grad():
            for s in tqdm(dataloader, desc="Generating Predictions"):
                s = [f.to(self.device) for f in s]
                score = self.compute_score(s[0])
                scores += score.squeeze().tolist()
                labels += s[1].squeeze().tolist()
                
        return scores, labels