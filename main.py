import argparse
from RunServer import run_server
from RunClient import run_client
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--cid", type=int)
    parser.add_argument("--algo", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold"])
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.algo, args.mu, args.alpha)
    elif args.mode == "client":
        if args.cid is None:
            raise ValueError("Client ID (--cid) required for client mode")
        run_client(args.cid, args.algo, args.mu, args.alpha)

