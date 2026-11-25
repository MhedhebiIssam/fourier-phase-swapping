import math

import matplotlib.pyplot as plt
import torch


class ImageViewer:
    def __init__(self, images=None, nrow=4, figsize=None, normalize=False):
        self.images = []
        self.titles = []
        self.nrow = nrow
        self.figsize = figsize
        self.normalize = normalize

        if images is not None:
            self.add(*images)

    def add(self, *images, titles=None):
        if titles is None:
            titles = [None] * len(images)
        elif isinstance(titles, str):
            titles = [titles]

        if len(titles) != len(images):
            raise ValueError("Each image must have exactly one title.")

        for img, title in zip(images, titles):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().float()
            else:
                from torchvision.transforms.functional import to_tensor

                img = to_tensor(img)

            # Handle batch: NCHW
            if img.ndim == 4:
                batch_titles = (
                    [None] * len(img)
                    if title is None
                    else [f"{title} {i}" for i in range(len(img))]
                )
                self.add(*img, titles=batch_titles)
                continue

            # Handle grayscale: HW
            if img.ndim == 2:
                img = img.unsqueeze(0)

            # Handle HWC -> CHW
            elif (
                img.ndim == 3
                and img.shape[-1] in (1, 3, 4)
                and img.shape[0] not in (1, 3, 4)
            ):
                img = img.permute(2, 0, 1)

            if img.ndim != 3:
                raise ValueError(f"Expected CHW image, got {img.shape}")

            self.images.append(img.contiguous())
            self.titles.append(title or f"Image {len(self.images)}")

        return self

    def show(self, images=None, titles=None):
        if images is not None:
            old_images = self.images
            old_titles = self.titles

            self.images = []
            self.titles = []
            self.add(*images, titles=titles)

        if not self.images:
            raise ValueError("No images to display.")

        n = len(self.images)
        cols = min(self.nrow, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=self.figsize or (cols * 3, rows * 3),
            squeeze=False,
        )

        for i, ax in enumerate(axes.flat):
            if i >= n:
                ax.axis("off")
                continue

            img = self.images[i]

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3]
            elif img.shape[0] != 3:
                raise ValueError(f"Unsupported channels: {img.shape[0]}")

            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(min(img.shape[1], 512), min(img.shape[2], 512)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            img = img.permute(1, 2, 0).numpy()

            ax.imshow(img)
            ax.set_title(self.titles[i])
            ax.axis("off")

        fig.suptitle(f"{n} Images")
        plt.tight_layout()
        plt.show()

        if images is not None:
            self.images = old_images
            self.titles = old_titles

        return fig, axes
