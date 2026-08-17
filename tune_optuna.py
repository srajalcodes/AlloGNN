# tune_optuna.py
import os
import json
import logging
import datetime
import torch
import optuna
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.data.dataset import AlloDataset
from src.models.allognn import AlloGNN
from src.training.trainer import AlloGNNTrainer

# Setup minimal logging to prevent console clutter during long runs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Define the Bayesian Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    pos_weight = trial.suggest_float("pos_weight", 5.0, 20.0)
    beta = trial.suggest_float("beta", 0.05, 0.5)
    
    # 2. Load Datasets (num_workers=0 is required for Windows HDF5 safety)
    train_dataset = AlloDataset(
        graph_dir="data/processed/graphs",
        emb_path="data/processed/embeddings/esm2_embeddings.h5",
        split_file="data/processed/splits/train_ids.txt"
    )
    val_dataset = AlloDataset(
        graph_dir="data/processed/graphs",
        emb_path="data/processed/embeddings/esm2_embeddings.h5",
        split_file="data/processed/splits/val_ids.txt"
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 3. Initialize Model and Optimizers
    model = AlloGNN().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Run a shorter 15-epoch trial to keep the optimization fast
    max_epochs = 15
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])

    trainer = AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=pos_weight, gamma=gamma, alpha=1.0, beta=beta
    )

    # 4. Short Training Loop
    best_trial_aupr = 0.0
    for epoch in range(1, max_epochs + 1):
        _ = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, epoch)
        
        # Target metric: Type IV (distal) AUPR
        type4_aupr = metrics.get('aupr_type4', 0.0)
        
        if type4_aupr > best_trial_aupr:
            best_trial_aupr = type4_aupr
            
        # Report progress to Optuna for potential pruning of bad trials
        trial.report(type4_aupr, epoch)
        if trial.should_prune():
            train_dataset.close()
            val_dataset.close()
            raise optuna.exceptions.TrialPruned()

    train_dataset.close()
    val_dataset.close()
    
    return best_trial_aupr

def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    db_path = "sqlite:///results/tables/optuna_study.db"
    
    # Use Tree-structured Parzen Estimator (TPE) for Bayesian Search
    study = optuna.create_study(
        study_name="allognn_hyperparam_tuner",
        storage=db_path,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5) # Prune bad runs after epoch 5
    )
    
    logging.info("Starting Bayesian Optimization Study...")
    logging.info(f"Database Storage: {db_path}")
    
    # Run 50 trials (approx. 6-8 hours total on your RTX A4500)
    # Increase n_trials to 100 or 150 if you want a deeper 24-48 hour run
    study.optimize(objective, n_trials=50)
    
    logging.info("\n=== STUDY COMPLETE ===")
    logging.info(f"Best Trial Parameter Configuration:")
    for k, v in study.best_params.items():
        logging.info(f"  {k}: {v}")
    logging.info(f"Best achieved Type IV AUPR: {study.best_value:.4f}")
    
    # Export best parameters to a JSON configuration file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = Path("results/tables") / f"best_hyperparameters_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump({
            "best_params": study.best_params,
            "best_val_type4_aupr": study.best_value,
            "timestamp": timestamp
        }, f, indent=4)
    logging.info(f"Optimized hyperparameters successfully exported to {config_path}")

if __name__ == "__main__":
    main()