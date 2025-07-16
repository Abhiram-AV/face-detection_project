# Fine-Tuning a Face Recognition Model on Low-Quality Data

This project demonstrates the process of taking a state-of-the-art pre-trained face recognition model (ArcFace from InsightFace), evaluating its baseline performance on a custom low-quality dataset, fine-tuning it, and then re-evaluating to measure the improvement.

## 🚀 Project Structure

- **/data**: Contains the image dataset. This directory is not checked into version control.
- **/src**: All Python source code modules.
  - `config.py`: Central configuration for paths, hyperparameters, etc.
  - `data_loader.py`: Handles dataset loading, splitting, and preprocessing.
  - `model.py`: Manages the face recognition model.
  - `evaluator.py`: Logic for running evaluations and calculating metrics.
  - `trainer.py`: The fine-tuning script.
  - `utils.py`: Helper functions, primarily for plotting.
- **/scripts**: Executable scripts to run the different stages of the project.
- **/outputs**: Directory to save model checkpoints, plots, and evaluation results.

## 📋 How to Run

1.  **Setup**: Place your dataset in the `data/images` folder following the structure `person_name/quality/image.jpg`.
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Prepare Dataset Splits**:
    ```bash
    python scripts/prepare_dataset.py
    ```
4.  **Run process row images**:
    ```bash
    python scripts/process_row_images.py
    ```
    This will save processed images to the `outputs/processed_images` directory.

5.  **Run Baseline Evaluation**:
    ```bash
    python scripts/run_baseline_eval.py
    ```
    This will save plots and metrics to the `outputs/baseline` directory.
6.  **Fine-Tune the Model**:
    ```bash
    python scripts/run_finetuning.py
    ```
    Model checkpoints will be saved in `outputs/checkpoints`.
7.  **Run Final Evaluation**:
    ```bash
    python scripts/run_final_eval.py
    ```
    This will save the final evaluation results to the `outputs/fine_tuned` directory and generate a comparison plot.

## ⚡ CUDA + ONNX Runtime GPU Setup (Optional but Recommended for GPU Acceleration)
To accelerate embedding extraction and face detection using your GPU, ensure CUDA and ONNX Runtime GPU are correctly installed:

✅ Step 1: Check if you have a compatible NVIDIA GPU
Run: nvidia-smi
You should see your GPU info and driver version.

✅ Step 2: Install CUDA Toolkit (required for ONNX GPU)
Download and install CUDA Toolkit 12.2

Choose:
OS: Windows 10 or 11
Architecture: x86_64
Installer: Local Installer (EXE)

After installation, add this to your System Environment PATH:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin

Restart your terminal or IDE.

✅ Step 3: Install ONNX Runtime GPU
pip install onnxruntime-gpu==1.15.1

## 📦 Requirements
Python 3.8+
PyTorch
ONNX Runtime (GPU optional)
InsightFace
OpenCV, NumPy, tqdm, matplotlib, etc
