import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device) # cpu