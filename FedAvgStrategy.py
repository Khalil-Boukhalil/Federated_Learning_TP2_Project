import flwr as fl
import numpy as np
import os
import json
from datetime import datetime
from flwr.common import Parameters
from typing import List, Tuple
from config import *
class FedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Parameters, save_name: str, mu: float = 0.0, alpha: float = 1.0):
        super().__init__(
            initial_parameters=initial_parameters,
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_FIT,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=lambda metrics: {
                "val_accuracy": float(np.mean([m[1].get("val_accuracy", 0.0) for m in metrics]))
            }
        )

        self.mu = mu
        self.alpha = alpha
        self.save_name = save_name
        self.history = {
            "metrics_distributed_fit": [],
            "metrics_distributed_eval": [],
            "rounds": []
        }


    def aggregate_fit(self, server_round, results, failures):
        for failure in failures:
            print(f"❌ Client failure: {failure}")

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if len(results) > 0:
            total = sum(res.num_examples for _, res in results)
            avg_loss = sum(res.metrics.get("train_loss", 0.0) * res.num_examples for _, res in results) / total
            self.history["metrics_distributed_fit"].append((server_round, avg_loss))
            self.history["rounds"].append(server_round)

        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if len(results) > 0:
            accuracies = [res.metrics.get("val_accuracy", 0.0) for _, res in results]
            avg_accuracy = float(np.mean(accuracies))
            self.history["metrics_distributed_eval"].append((server_round, avg_accuracy))
        return aggregated_loss, metrics

    

    def save_results(self):
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"results_{self.save_name}_mu_{self.mu}_alpha_{self.alpha}_{timestamp}.json"
        filepath = os.path.join("results", filename)

        results = {
            "rounds": [r for r, _ in self.history["metrics_distributed_fit"]],
            "loss": [loss for _, loss in self.history["metrics_distributed_fit"]],
            "val_accuracy": [acc for _, acc in self.history["metrics_distributed_eval"]],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {filepath}")

