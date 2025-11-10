import pandas as pd, joblib, argparse, os
from pathlib import Path
from utils import clean_dataframe, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Load a trained model and test it on a new dataset")
    parser.add_argument('--model', required=True, help="Model name used during training (lreg, rf, svr, sgd)")
    parser.add_argument('--new_data', required=True, help="Path to the new dataset for evaluation")
    parser.add_argument('--save_preds', action='store_true', help="If set, save predictions to CSV")
    args = parser.parse_args()

    model_name = args.model.lower()
    model_path = f"data/models/{model_name}_pipeline.joblib"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    new_df = pd.read_csv(args.new_data, low_memory=False)
    new_df = new_df.drop(columns=['Tipo_de_Navegação_da_Atracação_Apoio Marítimo'])
    X_new, y_new = clean_dataframe(new_df)

    print(f"\nEvaluating {model_name.upper()} on new dataset:")
    rmse, mae, r2 = evaluate_model(model, X_new, y_new, label="New Dataset")

    if args.save_preds:
        preds = model.predict(X_new)
        os.makedirs('data/predictions', exist_ok=True)
        out_path = f"data/predictions/{model_name}_preds.csv"
        pd.DataFrame({'y_true': y_new, 'y_pred': preds}).to_csv(out_path, index=False)
        print(f"Predictions saved to: {out_path}")


if __name__ == '__main__':
    main()
