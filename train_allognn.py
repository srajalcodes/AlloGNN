# train_allognn.py
import os
import json
import time
import torch
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.data.dataset import AlloDataset
from src.models.allognn import AlloGNN
from src.training.trainer import AlloGNNTrainer

def create_directory_structure():
    dirs = [
        Path("output/logs"),
        Path("output/checkpoints"),
        Path("results/plots"),
        Path("results/tables")
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def setup_logger(timestamp):
    log_file = Path("output/logs") / f"train_{timestamp}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_file

def save_training_plots(history, timestamp):
    epochs = history["epoch"]
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Total Loss", color='black', linewidth=2)
    plt.plot(epochs, history["bce"], label="BCE (Binding)", color='blue', linestyle='--')
    plt.plot(epochs, history["ce"], label="CE (Site Type)", color='red', linestyle=':')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["aupr_overall"], label="Overall AUPR", color='green', linewidth=2)
    plt.plot(epochs, history["aupr_type3"], label="Type III (Proximal Allo)", color='purple', linestyle='--')
    plt.plot(epochs, history["aupr_type4"], label="Type IV (Distal Allo)", color='orange', linestyle='-.')
    plt.xlabel("Epoch")
    plt.ylabel("AUPR Score")
    plt.title("Validation AUPR by Site Type")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path("results/plots") / f"metrics_plot_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Performance plots successfully saved to {plot_path}")

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directory_structure()
    log_file = setup_logger(timestamp)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("=== Starting AlloGNN Production Pipeline ===")
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Hardware: {device}")
    
    logging.info("Loading Datasets...")
    train_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/train_ids.txt")
    val_dataset = AlloDataset("data/processed/graphs", "data/processed/embeddings/esm2_embeddings.h5", "data/processed/splits/val_ids.txt")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    logging.info(f"Train Dataset Size: {len(train_dataset)} | Val Dataset Size: {len(val_dataset)}")
    logging.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 1. OPTUNA OPTIMIZED HYPERPARAMETERS
    opt_lr = 0.0008070161049782942
    opt_wd = 1.2243130080634888e-06
    opt_gamma = 1.0631476498675683
    opt_pos_weight = 8.264681596629156
    opt_beta = 0.0654674015416209

    model = AlloGNN().to(device)
    optimizer = AdamW(model.parameters(), lr=opt_lr, weight_decay=opt_wd)
    
    # Advanced Scheduler: Warmup for 5 epochs, then Cosine decay
    max_epochs = 75
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    # Instantiate Trainer with optimized parameters
    trainer = AlloGNNTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=device,
        pos_weight=opt_pos_weight, gamma=opt_gamma, alpha=1.0, beta=opt_beta
    )

    patience = 15
    epochs_no_improve = 0
    best_aupr_overall = 0.0
    best_aupr_allo = 0.0  # Monitored on Type IV (ALLO)
    
    history = {
        "epoch": [], "loss": [], "bce": [], "ce": [], 
        "aupr_overall": [], "aupr_type3": [], "aupr_type4": []
    }
    
    start_time = time.time()
    logging.info("Training loop initiated with Optuna-optimized hyperparameters.")
    
    # 2. Execute Training Loop
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        loss, bce, ce = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, epoch)
        
        epoch_time = time.time() - epoch_start
        
        aupr_all = metrics.get('aupr_overall', 0.0)
        aupr_t3 = metrics.get('aupr_type3', 0.0)
        aupr_t4 = metrics.get('aupr_type4', 0.0)
        
        logging.info(
            f"Ep {epoch:02d}/{max_epochs} | L: {loss:.4f} (BCE:{bce:.3f} CE:{ce:.3f}) | "
            f"AUPR_all: {aupr_all:.4f} | T3_AUPR: {aupr_t3:.4f} | T4_AUPR: {aupr_t4:.4f} | {epoch_time:.1f}s"
        )
        
        history["epoch"].append(epoch)
        history["loss"].append(float(loss))
        history["bce"].append(float(bce))
        history["ce"].append(float(ce))
        history["aupr_overall"].append(float(aupr_all))
        history["aupr_type3"].append(float(aupr_t3))
        history["aupr_type4"].append(float(aupr_t4))
        
        # Checkpointing
        if aupr_all > best_aupr_overall:
            best_aupr_overall = aupr_all
            torch.save(model.state_dict(), Path("output/checkpoints") / f"best_overall_{timestamp}.pt")
            
        # Monitor Type IV directly (the true allosteric blind spot)
        if aupr_t4 > best_aupr_allo:
            best_aupr_allo = aupr_t4
            torch.save(model.state_dict(), Path("output/checkpoints") / f"best_allo_{timestamp}.pt")
            logging.info(f"   -> New best ALLO model saved! (Type IV AUPR: {aupr_t4:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"\nEarly stopping triggered after {epoch} epochs. No Type IV AUPR improvement for {patience} epochs.")
                break

    total_time = (time.time() - start_time) / 60
    logging.info(f"\nTraining completed in {total_time:.2f} minutes.")
    
    save_training_plots(history, timestamp)

    csv_path = Path("results/tables") / f"run_history_{timestamp}.csv"
    pd.DataFrame(history).to_csv(csv_path, index=False)
    logging.info(f"Training history saved to CSV: {csv_path}")

    run_summary = {
        "timestamp": timestamp,
        "hardware": str(device),
        "hyperparameters": {
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": opt_lr,
            "weight_decay": opt_wd,
            "focal_gamma": opt_gamma,
            "pos_weight": opt_pos_weight,
            "loss_alpha_bce": 1.0,
            "loss_beta_ce": opt_beta
        },
        "results": {
            "epochs_completed": len(history["epoch"]),
            "total_time_minutes": total_time,
            "best_overall_aupr": best_aupr_overall,
            "best_allo_type4_aupr": best_aupr_allo,
            "final_type4_aupr": history["aupr_type4"][-1]
        }
    }
    
    json_path = Path("results/tables") / f"run_log_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(run_summary, f, indent=4)
    logging.info(f"Run profile saved to JSON: {json_path}")

    train_dataset.close()
    val_dataset.close()
    logging.info("All resources safely closed. Pipeline execution finished.")

if __name__ == "__main__":
    main()