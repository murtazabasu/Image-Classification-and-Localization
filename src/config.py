import os
import torch

# define root path 
ROOT_PATH = os.path.abspath(os.getcwd())[:-4]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_SIZE = 0.2