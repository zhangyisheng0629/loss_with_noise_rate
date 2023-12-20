
import torch
import os


noise_idx=torch.load(os.path.join("./results/select/cifar-10", "noise_idx.pt"))

print(noise_idx.tolist())