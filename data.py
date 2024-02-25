from torch.utils.data import Dataset
import torch
from torchvision import transforms


class SCEImageDataset(Dataset):
    def __init__(self, dataset, transform, triplet=False):
        super(self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.triplet = triplet

    def __getitem__(self, idx):
        N = len(self.dataset*2)
        xa_i = idx  # uniform[1,...,N]
        xa_i, _ = self.dataset[xa_i]

        j = torch.randint(low=0, high=N**2, size=(1,)).item()
        xr_i, xr_j = (j - 1) // N + 1 , (j - 1) % N + 1  # uniform[1,...N]^2

        if self.triplet: xr_i = xa_i  # reuse xa_i U[1,...N] and sample xr_i U[1,...N]

        xr_i, _ = self.dataset[xr_i]
        xr_j, _ = self.dataset[xr_j]

        xa_i, xa_j = self.transform(xa_i), self.transform(xa_i)
        xr_i, xr_j = self.transform(xr_i), self.transform(xr_j)
        return (xa_i, xa_i), (xa_j, xr_i)


class SSLImageDataset(Dataset):
    def __init__(self, dataset, transform):
        super(self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        xa_i, xa_j = self.transform(x, num_views=2)
        return xa_i, xa_j


# SimCLR augmentations
class SCECLRAugmentation():
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="train",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "train" or mode=="eval":
            augmentations = [
                transforms.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                transforms.RandomHorizontalFlip(),
            ]
        else:  # Test
            augmentations = [transforms.Resize(size=imgsize)]

        if mode == "train":
            augmentations+= [
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8*jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]

        augmentations += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.augmentations = transforms.Compose(augmentations)

    def __call__(self, x, num_views):
        return [self.augmentations(x) for _ in range(num_views) ] if num_views > 1 else self.augmentations(x)


