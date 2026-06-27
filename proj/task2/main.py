import os
import subprocess

# ═══════════════════════════════════════════════════════════════
#  WINNING DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# Hardcode your winning combination right here
WINNING_DATASETS = ["main_dataset", "extra-3", "custom", "extra-6"]

multiple = False
COMBINATIONS = [
    ["main_dataset", "extra-3", "custom"],
]

# 'model' e/ou 'marinho. Aplicar por esta orderm
PRE_PROCESSING = ["model"]

# preprocessing = ["addoc"]

def run_final_pipeline():
    print(f"🚀 Starting final pipeline run with datasets: {WINNING_DATASETS}\n")
    
    # 1. Set the environment variable so parser.py and task2.py can read it
    os.environ["ACTIVE_DATASETS"] = ",".join(WINNING_DATASETS)
    os.environ["PRE_PROCESSING"] = ",".join(PRE_PROCESSING)
    
    try:
        # 2. Run the Parser to build the final JSON catalog
        print("▶️  Running parser.py...")
        subprocess.run(["python", "../datasets/task2/parser.py"], check=True)

        print("▶️  Running pre_processor.py...")
        subprocess.run(["python", "pre_processor.py"], check=True)
        
        # 3. Run Task 2 to train and evaluate
        print("▶️  Running task2.py...")
        subprocess.run(["python", "task2.py"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Pipeline crashed!")
        print(f"Error details: {e}")
        return
        
    print("\n✅ PIPELINE COMPLETED! Your final model and results are ready.")

if __name__ == "__main__":
    if multiple:
        for combo in COMBINATIONS:
            WINNING_DATASETS = combo
            run_final_pipeline()
    else:
        run_final_pipeline()