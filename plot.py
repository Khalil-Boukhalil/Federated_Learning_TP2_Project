import os
import json
import re
import matplotlib.pyplot as plt

results_dir = "results"  

all_results = {"fedavg": {}, "fedprox": {}, "scaffold": {}}
skipped_files = []

for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        try:
            with open(os.path.join(results_dir, filename), "r") as f:
                data = json.load(f)

            rounds = data.get("rounds", [])
            loss = data.get("loss", [])
            acc = data.get("val_accuracy", [])

            if not (len(rounds) == len(loss) == len(acc)):
                skipped_files.append(filename)
                print(f"⚠️ Skipping invalid file (length mismatch): {filename}")
                continue

            lower = filename.lower()
            if "fedavg" in lower:
                strategy = "fedavg"
            elif "fedprox" in lower:
                strategy = "fedprox"
            elif "scaffold" in lower:
                strategy = "scaffold"
            else:
                skipped_files.append(filename)
                print(f"⚠️ Unknown strategy in filename: {filename}")
                continue

            alpha_match = re.search(r"alpha[_=-]?([0-9.]+)", lower)
            mu_match = re.search(r"mu[_=-]?([0-9.]+)", lower)

            alpha = alpha_match.group(1) if alpha_match else "?"
            mu = mu_match.group(1) if mu_match else "?"

            if strategy == "fedprox" or strategy == "scaffold":
                label = f"α={alpha}, μ={mu}"
            else:
                label = f"α={alpha}"


            all_results[strategy][label] = {
                "rounds": rounds,
                "loss": loss,
                "val_accuracy": acc
            }

        except Exception as e:
            skipped_files.append(filename)
            print(f"⚠️ Error reading file {filename}: {e}")

for strategy, configs in all_results.items():
    if not configs:
        continue

    plt.figure(figsize=(10, 6))
    for label, result in configs.items():
        plt.plot(result["rounds"], result["val_accuracy"], label=label)
    plt.title(f"Validation Accuracy - {strategy.upper()}")
    plt.xlabel("Rounds")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{strategy}_val_accuracy.png")
    plt.close()

plt.figure(figsize=(12, 7))
for strategy, configs in all_results.items():
    for label, result in configs.items():
        plt.plot(result["rounds"], result["val_accuracy"], label=f"{strategy.upper()} - {label}")
plt.title("Validation Accuracy - All Strategies")
plt.xlabel("Rounds")
plt.ylabel("Validation Accuracy")
plt.legend(fontsize=7)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_strategies_val_accuracy.png")
plt.close()

for strategy, configs in all_results.items():
    if not configs:
        continue

    plt.figure(figsize=(10, 6))
    for label, result in configs.items():
        plt.plot(result["rounds"], result["loss"], label=label)
    plt.title(f"Training Loss - {strategy.upper()}")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{strategy}_loss.png")
    plt.close()

plt.figure(figsize=(12, 7))
for strategy, configs in all_results.items():
    for label, result in configs.items():
        plt.plot(result["rounds"], result["loss"], label=f"{strategy.upper()} - {label}")
plt.title("Training Loss - All Strategies")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.legend(fontsize=7)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_strategies_loss.png")
plt.close()

print("✅ All plots saved successfully.")
