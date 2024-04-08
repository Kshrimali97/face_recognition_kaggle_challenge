## Face Recognition Model Training

This repository contains the code to train a face recognition model on a subset of the Celebrity Face Dataset.

### Usage

1. **Training and Validation (and testing):**
   - Run the script `train_val.py` to train and validate the model on the `train` dataset. PyTorch's built-in train/validation data split is used. Additionally the testing utility on the trained model is also added at the end of the script
     ```bash
     python3 train_val.py
     ```
All helper functions are contained within the scripts, allowing for standalone execution.

