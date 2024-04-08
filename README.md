## Face Recognition Model Training

This repository contains the code to train a face recognition model on a subset of the Celebrity Face Dataset.

### Usage

1. **Training and Validation:**
   - Run the script `train_val.py` to train and validate the model on the `train` dataset. PyTorch's built-in train/validation data split is used.
     ```bash
     python3 train_val.py
     ```

2. **Testing:**
   - Use the script `test.py` to test the trained model on the `test` dataset.
     ```bash
     python3 test.py
     ```

All helper functions are contained within the scripts, allowing for standalone execution.

