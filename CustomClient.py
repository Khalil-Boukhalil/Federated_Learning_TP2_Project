from typing import Dict, List, Optional
import copy
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from config import *
from typing import Tuple
class CustomClient(fl.client.NumPyClient):
    def __init__(self, cid: int, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device, algo: str,
                 mu: float = 0.0, global_c: Optional[List[torch.Tensor]] = None):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.algo = algo
        self.mu = mu
        self.global_c = global_c
        self.c_local = None
        self.initialized = False

    def get_parameters(self, config: Optional[Dict] = None) -> List[np.ndarray]:
        return self.model.get_model_parameters()

    def fit(self, parameters, config):
        print(f"✅ Client {self.cid} received config keys: {list(config.keys())}")
        try:
            initial_weights = copy.deepcopy(parameters) if self.algo == "scaffold" else None
            control_variate_diff = None

            if self.algo == "scaffold":
                self.global_c = [torch.tensor(c, dtype=torch.float32) for c in eval(config["c_global"])]
                if self.c_local is None:
                    self.c_local = [torch.zeros_like(g) for g in self.global_c]
                control_variate_diff = [c_l - c_g for c_l, c_g in zip(self.c_local, self.global_c)]

            self.model.set_model_parameters(parameters)
            self.model.to(self.device)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=config.get("learning_rate", LEARNING_RATE))
            criterion = nn.CrossEntropyLoss()

            for _ in range(config.get("epochs", EPOCHS)):
                self.model.train()
                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(x)
                    loss = criterion(output, y)

                    # ✅ SCAFFOLD correction (if scaffold is used)
                    if control_variate_diff is not None:
                        for p, cv in zip(self.model.parameters(), control_variate_diff):
                            if p.grad is not None:
                                p.grad += cv.to(self.device)

                    # ✅ FedProx regularization term
                    if self.algo == "fedprox" and self.mu > 0.0:
                        proximal_term = 0.0
                        for param, global_param in zip(self.model.parameters(), parameters):
                            global_tensor = torch.tensor(global_param, dtype=param.dtype, device=param.device)
                            proximal_term += ((param - global_tensor) ** 2).sum()
                        loss += (self.mu / 2) * proximal_term

                    loss.backward()
                    optimizer.step()


            if self.algo == "scaffold" and initial_weights is not None:
                new_weights = self.model.get_model_parameters()
                lr = config.get("learning_rate", LEARNING_RATE)
                steps = len(self.train_loader) * config.get("epochs", EPOCHS)
                for i, (w_new, w_old) in enumerate(zip(new_weights, initial_weights)):
                    delta = (torch.tensor(w_old) - torch.tensor(w_new)) / (lr * steps)
                    self.c_local[i] = self.c_local[i] - self.global_c[i] + delta

            train_loss = loss.item() if 'loss' in locals() else float("inf")
            metrics = {"train_loss": train_loss}

            if self.algo == "scaffold" and self.c_local is not None:
                metrics["c_local"] = str([c.tolist() for c in self.c_local])

            #print(f"📦 Metrics to return: {metrics}")
            return self.model.get_model_parameters(), len(self.train_loader.dataset), metrics

        except Exception as e:
            print(f"❌ [Client {self.cid}] fit() error: {e}")
            traceback.print_exc()
            return parameters, 0, {}
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        try:
            self.model.set_model_parameters(parameters)
            self.model.to(self.device)
            criterion = nn.CrossEntropyLoss()
            total_loss, correct, total = 0.0, 0, 0

            self.model.eval()
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    loss = criterion(output, y)
                    total_loss += loss.item() * y.size(0)
                    correct += output.argmax(1).eq(y).sum().item()
                    total += y.size(0)

            accuracy = correct / total
            print(f"   [Client {self.cid}] Evaluation accuracy: {accuracy:.4f}")
            return float(total_loss / total), total, {"val_accuracy": accuracy}
        except Exception as e:
            print(f"❌ [Client {self.cid}] Evaluation error: {str(e)}")
            traceback.print_exc()
            raise