from CustomFashionModel import CustomFashionModel
from config import *
import flwr as fl
from FedAvgStrategy import FedAvgStrategy
from ScaffoldStrategy import ScaffoldStrategy
from FedProxStrategy import FedProxStrategy
from flwr.common import ndarrays_to_parameters
def run_server(algo: str = "fedavg", mu: float = 0.0, alpha: float = 1.0) -> None:
    print(f"\n🚀 Starting {algo.upper()} server with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds")

    model = CustomFashionModel()
    initial_parameters = ndarrays_to_parameters(model.get_model_parameters())

    strategy_class = {
        "fedavg": FedAvgStrategy,
        "fedprox": FedProxStrategy,
        "scaffold": ScaffoldStrategy
    }[algo]

    strategy = strategy_class(initial_parameters, f"results_{algo}_alpha_{alpha}_mu_{mu}.json")

    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout= 60),
        strategy=strategy
    )
    strategy.save_results()
