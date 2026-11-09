# Standalone W&B Synchronization Script
# Reads existing saved metrics and logs to W&B without re-running training.
import os, sys, json, glob, argparse
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="pathmnist-fedavg-vs-fedprox")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    import wandb
    summary_file = os.path.join(PROJ_ROOT, "experiments", "results", "multiseed_consolidated_summary.json")
    with open(summary_file, "r") as f:
        data = json.load(f)

    for s_info in data["per_seed_results"]:
        seed = s_info["seed"]
        for algo in ["fedavg", "fedprox"]:
            run_name = s_info[algo]["run_name"]
            run = wandb.init(project=args.project, entity=args.entity, name=run_name, group=f"seed_{seed}", config=s_info[algo])
            # Log test metrics
            wandb.log({
                "test/accuracy": s_info[algo]["test_accuracy"],
                "test/macro_f1": s_info[algo]["test_macro_f1"],
                "test/balanced_accuracy": s_info[algo]["test_balanced_accuracy"],
                "test/loss": s_info[algo]["test_loss"],
                "val/best_accuracy": s_info[algo]["best_validation_accuracy"],
            })
            run.finish()
    print("Successfully synchronized 5 seeds to Weights & Biases!")

if __name__ == "__main__":
    main()
