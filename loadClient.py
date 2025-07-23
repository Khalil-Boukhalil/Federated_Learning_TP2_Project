import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple
import traceback
def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    try:
        images, labels = torch.load(os.path.join(data_dir, f"client_{cid}.pt"), weights_only=False)

        print(f"Client {cid} raw data shapes - Images: {images.shape if hasattr(images, 'shape') else 'unknown'}, Labels: {labels.shape if hasattr(labels, 'shape') else 'unknown'}")

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        images = images.float() 
        if images.ndim == 3:
            images = images.unsqueeze(1)

        print(f"Client {cid} processed data shapes - Images: {images.shape}, Labels: {labels.shape}")
        print(f"Unique labels: {torch.unique(labels)}")

        dataset = TensorDataset(images, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size)
        )
    except Exception as e:
        print(f"Error loading data for client {cid}: {str(e)}")
        traceback.print_exc()
        raise
