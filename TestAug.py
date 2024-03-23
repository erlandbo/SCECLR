from data import CIFARAugmentations
import torch

Augmentations = CIFARAugmentations

aug = Augmentations(imgsize=(32, 32), mean=(0.4914, 0.4822, 0), std=(0.2023, 0.1994, 0), num_views=2)
x = torch.rand(4, 3, 32, 32)
print(aug(x))