
### s6_eval_predict.py (Evaluation Script - Folder Mode)
#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np

def evaluate_file(path, hist_days=7):
    """
    Evaluate the prediction accuracy of a single predict_result_hisfut.csv file.
    The returned dict contains: filename, historical date range, WMAPE (subscription), WMAPE (redemption), and overall prediction accuracy.
    """
    df = pd.read_csv(path, dtype={'transaction_date': str})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')
    hist_df = df[df['apply_amt_actual'].notna()].copy()

    if not hist_df.empty:
        start = hist_df['transaction_date'].min().strftime('%Y%m%d')
        end   = hist_df['transaction_date'].max().strftime('%Y%m%d')
        date_range = f"{start} to {end}"
    else:
        date_range = 'N/A'
    wm_apply = {}
    sum_apply_real = {}
    wm_redeem = {}
    sum_redeem_real = {}
 
    for code, grp in hist_df.groupby('fund_code'):
        real_a = grp['apply_amt_actual'].astype(float)
        pred_a = grp['apply_amt_pred'].astype(float)
        mask_a = real_a > 0

        if mask_a.any():
            weights = real_a[mask_a] / real_a[mask_a].sum()
            wm_apply[code] = ((pred_a[mask_a] - real_a[mask_a]).abs() / real_a[mask_a] * weights).sum()
            sum_apply_real[code] = real_a[mask_a].sum()

        real_r = grp['redeem_amt_actual'].astype(float)
        pred_r = grp['redeem_amt_pred'].astype(float)
        mask_r = real_r > 0

        if mask_r.any():
            weights_r = real_r[mask_r] / real_r[mask_r].sum()
            wm_redeem[code] = ((pred_r[mask_r] - real_r[mask_r]).abs() / real_r[mask_r] * weights_r).sum()
            sum_redeem_real[code] = real_r[mask_r].sum()

    total_apply = sum(sum_apply_real.values()) if sum_apply_real else 0
    total_redeem = sum(sum_redeem_real.values()) if sum_redeem_real else 0

    wmape_apply = (sum(wm_apply[c] * sum_apply_real[c] for c in wm_apply) / total_apply) if total_apply > 0 else np.nan

    wmape_redeem = (sum(wm_redeem[c] * sum_redeem_real[c] for c in wm_redeem) / total_redeem) if total_redeem > 0 else np.nan

    combined_score = 0.5 * wmape_apply + 0.5 * wmape_redeem

    return {
        'file': os.path.basename(path),
        'date_range': date_range,
        'WMAPE_Subscription': wmape_apply,
        'WMAPE_Redemption': wmape_redeem,
        'rediction accuracy': combined_score
    }

def main():

    parser = argparse.ArgumentParser(description='Eval all predict_result_hisfut CSV files in a folder')
    parser.add_argument('--dir', default='./eval_predict', help='Folder with predict_result_hisfut CSVs')

    args = parser.parse_args()
    folder = args.dir
    pattern = os.path.join(folder, '*.csv')
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV files found in {folder}")
        return

    results = [evaluate_file(f) for f in files]
    results.sort(key=lambda x: x['Prediction accuracy'])

    print("\nPer-file evaluation:")

    for r in results:
        print(f"- {r['file']} | Range: {r['date_range']} | "
              f"WMAPE_subscription: {r['WMAPE_Subscription']:.4f} | WMAPE_Redemption: {r['WMAPE_Redemption']:.4f} | "
              f"Prediction accuracy: {r['Prediction accuracy']:.4f}")

    print("\nOverall ranking by Prediction accuracy (ascending):")
    print(f"{'File':<30} {'Date Range':<17} {'Score':>8}")

    for r in results:
        print(f"{r['file']:<30} {r['date_range']:<17} {r['Prediction accuracy']:>8.4f}")

if __name__ == '__main__':
    main()



 