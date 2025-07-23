from config import *
from FedAvgStrategy import FedAvgStrategy
from flwr.common import Parameters
import flwr as fl
class FedProxStrategy(FedAvgStrategy):
    def __init__(self, initial_parameters: Parameters, save_name: str, mu: float = 0.1):
        super().__init__(initial_parameters, save_name)
        self.mu = mu  # Proximal term coefficient

    def configure_fit(self, server_round, parameters, client_manager):
        """Override to send mu to clients."""
        fit_config = {
            "mu": self.mu,  # Pass mu to clients
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "server_round": server_round
        }
        clients = client_manager.sample(self.min_fit_clients, min_num_clients=self.min_fit_clients)
        return [(client, fl.common.FitIns(parameters, fit_config)) for client in clients]