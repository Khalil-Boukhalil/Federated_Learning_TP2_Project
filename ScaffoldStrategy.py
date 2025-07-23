from FedAvgStrategy import FedAvgStrategy
from config import *
import flwr as fl
import numpy as np
import traceback
from flwr.common import Parameters, parameters_to_ndarrays
class ScaffoldStrategy(FedAvgStrategy):
    def __init__(self, initial_parameters: Parameters, save_name: str):
        super().__init__(initial_parameters, save_name)
        self.global_c = None
        self.initialize_global_c(initial_parameters)

    def initialize_global_c(self, parameters: Parameters) -> None:
        try:
            if parameters is None:
                raise ValueError("No parameters provided for global_c initialization")

            ndarrays = parameters_to_ndarrays(parameters)
            self.global_c = [np.zeros_like(p) for p in ndarrays]
            print("✅ Initialized global_c with shapes:")
            for i, c in enumerate(self.global_c):
                print(f"  c_global {i}: {c.shape}")
        except Exception as e:
            print(f"❌ Failed to initialize global_c: {str(e)}")
            traceback.print_exc()
            raise

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"\n🧠 SCAFFOLD Server Round {server_round} - Preparing FIT config")

        if self.global_c is None:
            self.initialize_global_c(parameters)
            if self.global_c is None:
                raise RuntimeError("Global control variates not initialized")

        fit_config = {
            "c_global": str([c.tolist() for c in self.global_c]),
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "server_round": server_round
        }

        sample_size = min(self.min_fit_clients, client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients=sample_size)

        print(f"Selected {len(clients)} clients for round {server_round}")
        return [(client, fl.common.FitIns(parameters, fit_config)) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        for failure in failures:
            print(f"❌ Client failure in round {server_round}: {failure}")

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if len(results) > 0 and self.global_c is not None:
            new_c_list = []
            for client, res in results:
                if "c_local" in res.metrics:
                    try:
                        client_c = [np.array(c) for c in eval(res.metrics["c_local"])]
                        if len(client_c) == len(self.global_c):
                            new_c_list.append(client_c)
                            print(f"Received c_local from client {client.cid}")
                        else:
                            print(f"⚠️ Shape mismatch in c_local from client {client.cid}")
                    except Exception as e:
                        print(f"❌ Error processing c_local from client {client.cid}: {str(e)}")
                        continue

            if new_c_list:
                print(f"Updating global_c with {len(new_c_list)} client updates")
                for i in range(len(self.global_c)):
                    try:
                        delta_c = np.mean([c[i] - self.global_c[i] for c in new_c_list], axis=0)
                        self.global_c[i] += delta_c / len(new_c_list)
                    except Exception as e:
                        print(f"❌ Error updating global_c[{i}]: {str(e)}")
                        traceback.print_exc()

        return aggregated_parameters, metrics
