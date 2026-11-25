import random
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms.functional import (
    pil_to_tensor,
    rgb_to_grayscale,
    to_pil_image,
    to_tensor,
)

from Operations import fft, ifft
from Plotter import ImageViewer

data_folder = Path(__file__).resolve().parent.parent / "data" / "cifar100"
data_folder.mkdir(parents=True, exist_ok=True)

cifar100_folder = data_folder / "cifar100"
download = not cifar100_folder.exists()


train_data = CIFAR100(
    root=data_folder, train=True, download=download, transform=to_tensor
)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=2)

test_data = CIFAR100(
    root=data_folder, download=download, train=False, transform=to_tensor
)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

class_names = train_data.classes


viewer = ImageViewer(nrow=4)


def fuse_images():
    target = random.choice(train_data.targets)
    target_images = [tensor for tensor, label in train_loader if label == target]
    img_1 = random.choice(target_images)
    img_2 = random.choice(target_images)
    mg1, ph1 = fft(img_1)
    mg2, ph2 = fft(img_2)
    comb_01 = ifft(mg1, ph2).clamp(0, 1)
    comb_02 = ifft(mg2, ph1).clamp(0, 1)
    viewer.add(
        img_1,
        img_2,
        comb_01,
        comb_02,
        titles=[
            "Image 1",
            "Image 2",
            "Magnitude 1 + Phase 2",
            "Magnitude 2 + Phase 1",
        ],
    )


fuse_images()
fuse_images()
viewer.show()
