import torch
import torch.nn as nn
import numpy as np
from typing import List
class CustomFashionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def get_model_parameters(self) -> List[np.ndarray]:
        params = [val.cpu().numpy() for _, val in self.state_dict().items()]
        print("Model parameter shapes:")
        for i, p in enumerate(params):
            print(f"  Param {i}: {p.shape}")
        return params

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        if parameters is None:
            return
        state_dict = self.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            if isinstance(val, np.ndarray):
                state_dict[key] = torch.from_numpy(val)
            else:
                state_dict[key] = torch.tensor(val)
        self.load_state_dict(state_dict)
