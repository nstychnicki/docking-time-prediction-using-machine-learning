import pandas as pd, joblib, sys, argparse, os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from utils import clean_dataframe, evaluate_model, plot_train_val_errors, extract_feature_importance


def get_pipeline_and_params(model_name):
    if model_name == 'lreg':
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        params = {'model__fit_intercept': [True, False]}

    elif model_name == 'rf':
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=0, n_jobs=-1))
        ])
        params = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}

    elif model_name == 'svr':
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR())
        ])
        params = {'model__C': [0.1, 1, 10], 'model__epsilon': [0.01, 0.1], 'model__kernel': ['rbf', 'linear']}

    elif model_name == 'sgd':
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SGDRegressor(max_iter=3000, tol=1e-4, random_state=0))
        ])
        params = {'model__alpha': [1e-4, 1e-3, 1e-2], 'model__penalty': ['l2', 'l1']}

    else:
        sys.exit(f"Unknown model type: {model_name}")

    return pipeline, params


def main():
    parser = argparse.ArgumentParser(description="Train and save regression model")
    parser.add_argument('--model', required=True, help="Model: lreg, rf, svr, sgd")
    parser.add_argument('--data', default='data/sample_2024.csv', help="Path to main dataset")
    parser.add_argument('--verbose', default='false', choices=['true', 'false'], help="Output verbosity")
    args = parser.parse_args()

    model_name = args.model.lower()

    model_folder = os.path.join('data/models/', model_name)
    fi_folder = os.path.join(model_folder, 'feature_importance')
    plots_folder = os.path.join(model_folder, 'plots')

    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(fi_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)


    df = pd.read_csv(args.data, low_memory=False)
    X, y = clean_dataframe(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    pipeline, param_grid = get_pipeline_and_params(model_name)

    print(f"Running {model_name.upper()} with GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, scoring='r2', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    if args.verbose == 'true':
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(os.path.join(model_folder, 'grid_cv_verbosity.csv'))
        print(results[['params', 'mean_test_score', 'std_test_score']])

    print(f"\nBest Params: {grid.best_params_}")
    print("\nPerformance:")
    train_metrics = evaluate_model(best_model, X_train, y_train, "Train")
    val_metrics = evaluate_model(best_model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(best_model, X_test, y_test, "Test")

    plot_path = os.path.join(plots_folder, "train_val_errors.png")
    plot_train_val_errors(model_name, train_metrics, val_metrics, plot_path)
    print(f"Saved train/val error plot: {plot_path}")

    fi_df = extract_feature_importance(best_model, model_name, X_val, y_val)
    fi_path = os.path.join(fi_folder, f"{model_name}_importances.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved feature importances: {fi_path}")

    model_path = os.path.join(model_folder, f"{model_name}_pipeline.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved model: {model_path}")


if __name__ == '__main__':
    main()
