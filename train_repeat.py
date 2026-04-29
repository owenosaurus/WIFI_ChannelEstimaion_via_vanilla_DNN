import argparse
import os

import pandas as pd

from train import resolve_results_dir, train_one_snr


def run_snr_sweep(
    snr_list,
    data_dir: str = "/home/jinx/project/CE01/data_set",
    save_dir: str = "/home/jinx/project/CE01/results",
    csv_name: str = "results_table.csv",
):
    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    records = []

    for snr_db in snr_list:
        result = train_one_snr(
            snr_db=snr_db,
            data_dir=data_dir,
            save_dir=save_dir,
        )
        records.append((result["snr_db"], result["best_eval_nmae"]))

    records.sort(key=lambda item: item[0])
    sorted_snr_list = [snr_db for snr_db, _ in records]
    nmae_values = [best_eval_nmae for _, best_eval_nmae in records]

    df = pd.DataFrame({"DNN_NMAE": nmae_values})
    csv_path = os.path.join(results_dir, csv_name)
    df.to_csv(csv_path, index=False)

    print("\nsweep finished.")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"Row order (SNR_dB): {sorted_snr_list}")
    print(df)

    return df, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_list", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15, 18])
    parser.add_argument("--data_dir", type=str, default="/home/jinx/project/CE01/data_set")
    parser.add_argument("--save_dir", type=str, default="/home/jinx/project/CE01/results")
    parser.add_argument("--csv_name", type=str, default="results_table.csv")
    args = parser.parse_args()

    run_snr_sweep(
        snr_list=args.snr_list,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        csv_name=args.csv_name,
    )


if __name__ == "__main__":
    main()
