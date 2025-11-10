# Predicting Boat Docking Time with Traditional Machine Learning Algorithms 

This repository contains Python scripts for **training and testing Machine Learning models** for the task of predicting boat docking times. It supports Linear Regression, Random Forest, Support Vector Regression, and Stochastic Gradient Descent.

---

## Requirements

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib joblib
```

---

## 1. `train_model.py`

Trains a selected regression algorithm on the provided dataset, performs grid search hyperparameter tuning, evaluates performance, plots results, and saves the trained model and feature importances.

### **Usage:**

```bash
python train_model.py --model <model_name> [--data <path>] [--verbose <level>]
```

### **Arguments:**

| Flag        | Description                                                        | Default                 |
| ----------- | ------------------------------------------------------------------ | ----------------------- |
| `--model`   | Model to train (`lreg`, `rf`, `svr`, `sgd`)                        | *required*              |
| `--data`    | Path to training dataset                                           | `data/sample_2024.csv`  |
| `--verbose` | Verbosity level: `false` (summary) `true` (detailed CV results)    | `true`                  |

### **Example:**

```bash
python train_model.py --model rf --data data/sample_2024.csv --verbose true
```

### **Output Files:**

Files saved on the `./data/models/<model>/` folder are:

| File                                              | Description                               |
| ------------------------------------------------- | ----------------------------------------- |
| `<model>_pipeline.joblib`                         | Trained model saved via Joblib            |
| `<model>_importances.csv`                         | Feature importances or coefficients       |
| `<model>_train_val_errors.png`                    | Train vs Validation error comparison plot |

---

## 2. `test_model.py`

Loads a previously trained model and evaluates it on a new dataset. Optionally saves predictions.

### **Usage:**

```bash
python test_model.py --model <model_name> --new_data <path> [--save_preds]
```

### **Arguments:**

| Flag           | Description                                                            | Default    |
| -------------- | ---------------------------------------------------------------------- | ---------- |
| `--model`      | Name of the model used during training (`lreg`, `rf`, `svr`, `sgd`)    | *required* |
| `--new_data`   | Path to new dataset containing the same columns as training data       | *required* |
| `--save_preds` | If included, saves predictions to `data/predictions/<model>_preds.csv` | *off*      |

### **Example:**

```bash
python test_model.py --model rf --new_data data/sample_2025.csv --save_preds
```

### **Output Files:**

| File                                 | Description                                                 |
| ------------------------------------ | ----------------------------------------------------------- |
| `data/predictions/<model>_preds.csv` | CSV with true and predicted values (if `--save_preds` used) |

---

## Expected Workflow

1. **Train a model:**

   ```bash
   python train_model.py --model rf --data data/sample_2024.csv
   ```

2. **Test it on new data:**

   ```bash
   python test_model.py --model rf --new_data data/sample_2025.csv --save_preds
   ```

3. **Check results**

---
