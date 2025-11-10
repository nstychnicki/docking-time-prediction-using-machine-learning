import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.inspection import permutation_importance


def clean_dataframe(df: pd.DataFrame):
    for c in list(df.columns):
        if df[c].nunique() == len(df):
            df = df.drop(columns=[c])
            print(f"Dropped unique-id column: {c}")

    if 'TEstadia' not in df.columns:
        sys.exit("Error: 'TEstadia' target column not found.")

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c].astype(str)
                .str.replace(r'\\.', '', regex=True)
                .str.replace(',', '.', regex=True),
                errors='coerce'
            )

    df = df.dropna(subset=['TEstadia'])
    y = df['TEstadia'].astype(float)
    X = df.drop(columns=['TEstadia'])
    return X, y


def evaluate_model(model, X, y, label="Test"):
    preds = model.predict(X)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{label} RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return rmse, mae, r2


def plot_train_val_errors(model_name, train_metrics, val_metrics, out_path):
    metrics_names = ['RMSE', 'MAE', 'R²']
    plt.figure(figsize=(6,4))
    x = range(len(metrics_names))
    plt.bar([i-0.2 for i in x], train_metrics, width=0.4, label='Train')
    plt.bar([i+0.2 for i in x], val_metrics, width=0.4, label='Validation')
    plt.xticks(x, metrics_names)
    plt.title(f"{model_name.upper()} - Train vs Validation Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def extract_feature_importance(model, model_name, X_val, y_val):
    model_step = model.named_steps['model']
    if model_name == 'rf':
        importances = model_step.feature_importances_
    elif model_name in ['lreg', 'sgd']:
        importances = model_step.coef_
    else:  # SVR
        imp = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)
        importances = imp.importances_mean
    return pd.DataFrame({'feature': X_val.columns, 'importance': importances})
