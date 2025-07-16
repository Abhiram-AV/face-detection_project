import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trainer import run_finetuning

def main():
    best_model_path = run_finetuning()
    print(f"\n Fine-tuning finished. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()