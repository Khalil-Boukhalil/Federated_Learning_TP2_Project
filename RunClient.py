import os
import traceback
import torch
from CustomFashionModel import CustomFashionModel
from loadClient import load_client_data
from config import *
from CustomClient import CustomClient   
import flwr as fl
def run_client(cid: int, algo: str = "fedavg", mu: float = 0.0, alpha: float = 1.0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔌 [Client {cid}] Loading data and initializing {algo.upper()} client on {device}")

    try:
        data_path = os.path.join("client_data", f"alpha_{alpha}")
        train_loader, val_loader = load_client_data(cid, data_path, BATCH_SIZE)
        print(f"   [Client {cid}] Loaded {len(train_loader.dataset)} training samples from {data_path}")

        model = CustomFashionModel()
        global_c = None

        if algo == "scaffold":
            model_params = model.get_model_parameters()
            global_c = [torch.zeros_like(torch.tensor(p)) for p in model_params]
            print(f"   [Client {cid}] Initialized SCAFFOLD control variates with shapes:")
            for i, c in enumerate(global_c):
                print(f"  c_global {i}: {c.shape}")

        client = CustomClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            algo=algo,
            mu=mu,
            global_c=global_c
        )

        print(f"🔗 [Client {cid}] Connecting to server...")
        fl.client.start_client(
            server_address="127.0.0.1:8082",
            client=client.to_client(),
        )
    except Exception as e:
        print(f"❌ [Client {cid}] Initialization failed: {str(e)}")
        traceback.print_exc()
        raise
