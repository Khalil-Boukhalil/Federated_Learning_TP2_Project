import torch
import random
import numpy as np
SEED = 42
NUM_ROUNDS = 50
NUM_CLIENTS = 10
EPOCHS = 3
FRACTION_FIT = 1.0
BATCH_SIZE = 64
LEARNING_RATE = 0.01

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)