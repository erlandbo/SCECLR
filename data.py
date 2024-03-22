from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2


class SSLImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        xa_i, xa_j = self.transform(x)
        return xa_i, y, idx, xa_j

    def __len__(self):
        return len(self.dataset)


# SimCLR augmentations
class Augmentation():
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="train",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 num_views = 1
                 ):
        self.num_views = num_views
        if mode == "train" or mode=="eval":
            augmentations = [
                transforms.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                transforms.RandomHorizontalFlip(),
            ]
        else:  # Test
            augmentations = [transforms.Resize(size=imgsize)]

        if mode == "train":
            augmentations += [
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

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


class Augmentationv2():
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="train",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 num_views = 1
                 ):
        self.num_views = num_views
        if mode == "train" or mode=="eval":
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                v2.RandomHorizontalFlip(0.5),
            ]
        else:  # Test
            augmentations = [v2.Resize(size=imgsize)]

        if mode == "train":
            augmentations += [
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8*jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                v2.RandomGrayscale(p=0.2)
            ]

        augmentations += [
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),
        ]
        self.augmentations = v2.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


