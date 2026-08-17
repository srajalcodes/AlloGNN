# train.py
import sys
import time
import logging
from pathlib import Path

# Safe-guard python path for Windows execution
sys.path.append(str(Path(__file__).parent.resolve()))

# BUG FIXED: Import stages with correct fully-qualified pathing
from src.data import download as stage1
from src.data import preprocess as stage2
from src.data import esm2_embedder as stage3
from src.data import graph_builder as stage4
import add_surface_features as stage5
from src.data import create_splits as stage6
import train_allognn as stage7

# Master logging format
logging.basicConfig(
    level=logging.INFO,
    format="=== %(asctime)s [STAGE CONTROLLER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class ScriptArgs:
    """Mock argument parser for modules requiring namespace args."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_stage(stage_num, stage_name, main_func, args=None):
    """Executes a pipeline stage with timing and error handling."""
    logging.info(f"STARTING STAGE {stage_num}: {stage_name}")
    start_time = time.time()
    try:
        if args is not None:
            main_func(args)
        else:
            main_func()
        elapsed = (time.time() - start_time) / 60
        logging.info(f"STAGE {stage_num} COMPLETED SUCCESSFULLY in {elapsed:.2f} minutes.\n")
        return True
    except Exception as e:
        logging.error(f"FATAL ERROR IN STAGE {stage_num} ({stage_name}): {str(e)}")
        return False

def main():
    master_start = time.time()
    logging.info("Initializing Master Pipeline Orchestrator...")

    # Stage 1: RCSB PDB Structure Download
    download_args = ScriptArgs(base_dir=".", workers=16)
    if not run_stage(1, "RCSB Structure Download", stage1.main, download_args):
        return

    # Stage 2: Preprocess spatial coordinates & binding labels
    preprocess_args = ScriptArgs(
        meta_path="data/processed/splits/subset_metadata.csv",
        cif_dir="data/raw/kincore/cif",
        out_dir="data/processed/structures",
        cutoff=4.5
    )
    if not run_stage(2, "Spatial Preprocessing", stage2.main, preprocess_args):
        return

    # Stage 3: Generate ESM2 Sequence Embeddings
    if not run_stage(3, "ESM2 Embedding Extraction", stage3.main):
        return

    # Stage 4: Construct PyTorch Geometric Contact Graphs
    if not run_stage(4, "PyG Contact Graph Construction", stage4.main):
        return

    # Stage 5: Calculate SASA and Surface features
    if not run_stage(5, "Surface Geometry Calculation", stage5.main):
        return

    # Stage 6: Create Gene-Cold Splits (completely leakproof)
    if not run_stage(6, "Gene-Cold Split Generation", stage6.main):
        return

    # Stage 7: Train the Multi-Modal Gated AlloGNN
    if not run_stage(7, "AlloGNN Neural Network Training", stage7.main):
        return

    total_elapsed = (time.time() - master_start) / 60
    logging.info(f"ALL PIPELINE STAGES COMPLETED. TOTAL EXECUTION TIME: {total_elapsed:.2f} minutes.")

if __name__ == "__main__":
    main()