# src/training/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

class AlloGNNTrainer:
    def __init__(self, model, optimizer, scheduler, device, pos_weight=10.0, gamma=1.5, alpha=1.0, beta=0.15):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

    def focal_loss(self, preds, targets):
        """
        Focal Loss with dynamic gamma and pos_weight hyperparameters.
        """
        preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        
        bce_pos = -torch.log(preds)
        bce_neg = -torch.log(1.0 - preds)
        
        # Focal scaling: (1 - p)^gamma
        focal_pos = ((1.0 - preds) ** self.gamma) * bce_pos
        focal_neg = (preds ** self.gamma) * bce_neg
        
        # Label smoothing (5%)
        epsilon = 0.05
        targets_smooth = targets * (1.0 - epsilon) + epsilon * 0.5
        
        loss = self.pos_weight * targets_smooth * focal_pos + (1.0 - targets_smooth) * focal_neg
        return loss.mean()

    def loss_fn(self, outputs, data):
        preds = outputs['binding']
        targets = data.y.float()
        
        # Calculate Focal Loss using tuned parameters
        loss_binding = self.focal_loss(preds, targets)
        
        # Weakly-supervised auxiliary site type classification
        residue_types = data.site_type[data.batch]
        valid_mask = residue_types != -1
        
        if valid_mask.sum() > 0:
            loss_type = self.ce_loss(
                outputs['type'][valid_mask],
                residue_types[valid_mask]
            )
        else:
            loss_type = torch.tensor(0.0, device=self.device)
            
        total = self.alpha * loss_binding + self.beta * loss_type
        return total, loss_binding.item(), loss_type.item()

    def train_epoch(self, loader, epoch_idx):
        self.model.train()
        total_loss, total_bce, total_ce = 0, 0, 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx:02d} [Train]", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(batch)
            loss, bce, ce = self.loss_fn(outputs, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce
            total_ce += ce
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        self.scheduler.step()
        return total_loss/len(loader), total_bce/len(loader), total_ce/len(loader)

    @torch.no_grad()
    def evaluate(self, loader, epoch_idx):
        self.model.eval()
        all_preds, all_labels, all_types = [], [], []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch_idx:02d} [Eval]", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            all_preds.append(outputs['binding'].cpu())
            all_labels.append(batch.y.cpu())
            all_types.append(batch.site_type[batch.batch].cpu())
            
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        types = torch.cat(all_types).numpy()
        
        res = {}
        res['auroc_overall'] = roc_auc_score(labels, preds)
        res['aupr_overall'] = average_precision_score(labels, preds)
        
        tmap = {0:'type1', 1:'type1_5', 2:'type2', 3:'type3', 4:'type4'}
        for tid, tname in tmap.items():
            mask = (types == tid)
            if mask.sum() > 0 and labels[mask].sum() > 0:
                res[f'aupr_{tname}'] = average_precision_score(labels[mask], preds[mask])
                
        return res