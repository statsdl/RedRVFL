import argparse

from redrvfl.finance import download_dji, run_dji_paper_experiment


def parse_seeds(value):
    return tuple(int(seed.strip()) for seed in value.split(",") if seed.strip())


def main():
    parser = argparse.ArgumentParser(description="Run the paper-style DJI RedRVFL experiment.")
    parser.add_argument("--dataset-path", default="datasets/DJI.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, for example: 0,1,2")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--look-ahead", type=int, default=1)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=100)
    args = parser.parse_args()

    if args.download:
        download_dji(args.dataset_path)

    results = run_dji_paper_experiment(
        dataset_path=args.dataset_path,
        seeds=parse_seeds(args.seeds),
        horizon=args.horizon,
        look_ahead=args.look_ahead,
        n_layers=args.layers,
        max_evals=args.max_evals,
    )
    for result in results:
        print(
            f"{result.dataset} seed={result.seed} {result.model:7s} "
            f"RMSE={result.rmse:.6f} MAE={result.mae:.6f} MAPE={result.mape:.6f} "
            f"tune={result.tuning_seconds:.3f}s train={result.training_seconds:.3f}s "
            f"test={result.testing_seconds:.3f}s"
        )
        print(f"best_params={result.best_params}")


if __name__ == "__main__":
    main()
