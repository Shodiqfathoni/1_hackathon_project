import argparse, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.data_loader import load_csv, basic_checks
from src.preprocessing import build_preprocessor, get_feature_names, save_preprocessor
from src.model_builder import build_baseline_pipelines, build_stacking_pipeline
from src.evaluate import evaluate_model, cross_val_report
from src.utils import save_json
from src.utils import plot_actual_vs_pred, plot_model_comparison

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--target', type=str, default='emisi_CO2e')
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--model_dir', type=str, default='models')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_csv(args.data_path)
    checks = basic_checks(df)
    save_json(checks, Path(args.output_dir) / 'data_checks.json')
    print('data checks saved')

    # drop rows where target is missing
    df = df.dropna(subset=[args.target]).reset_index(drop=True)

    # define features (customize as needed)
    numeric_features = ['produksi_ton','solar_liter','listrik_kWh']
    categorical_features = []

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Build models
    baselines = build_baseline_pipelines(preprocessor)
    stack_pipe = build_stacking_pipeline(preprocessor)

    # Evaluate baselines
    results = {}
    for name, pipe in baselines.items():
        print('Training baseline:', name)
        res = evaluate_model(pipe, X_train, y_train, X_test, y_test)
        results[name] = res

    # Evaluate stacking
    print('Training stacking...')
    results['Stacking'] = evaluate_model(stack_pipe, X_train, y_train, X_test, y_test)

    # Cross validation for stacking
    cv_report = cross_val_report(stack_pipe, X, y, n_splits=5)
    results['Stacking_CV'] = cv_report

    # Save best stacking model (fit on full train for final artifact)
    stack_pipe.fit(X_train, y_train)
    joblib.dump(stack_pipe, Path(args.model_dir) / 'best_stacking_model.pkl')

        # simpan figures dir
    fig_dir = Path(args.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Plot perbandingan model (menggunakan results dict)
    # pilih metric; jika ingin r2_test:
    comp_path = fig_dir / "model_comparison_r2_test.png"
    plot_model_comparison(results, metric='r2_test', save_path=str(comp_path))
    print("Saved model comparison plot to", comp_path)

    # 2) Plot actual vs predicted untuk model stacking (setelah fit & predict)
    # Pastikan model sudah fit; gunakan X_test sebelumnya
    y_pred_stack = stack_pipe.predict(X_test)
    avp_path = fig_dir / "stacking_actual_vs_pred.png"
    plot_actual_vs_pred(y_test, y_pred_stack, save_path=str(avp_path), title="Stacking: Actual vs Pred")
    print("Saved actual vs pred plot to", avp_path)


    # Save results
    save_json(results, Path(args.output_dir) / 'results_summary.json')
    print('Training finished. Results saved to', args.output_dir)

if __name__ == '__main__':
    main()
