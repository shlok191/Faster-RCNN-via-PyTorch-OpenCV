from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import patches


def view(images, labels, k, std=1, mean=0):
    """Helper function that adds appropriate boxes around images!"""

    figure = plt.figure(figsize=(30, 30))

    images = list(images)
    labels = list(labels)

    for i in range(k):
        out = torchvision.utils.make_grid(images[i])
        inp = out.cpu().numpy().transpose([1, 2, 0])

        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)

        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))

        l = labels[i]["boxes"].cpu().numpy()
        l[:, 2] = l[:, 2] - l[:, 0]
        l[:, 3] = l[:, 3] - l[:, 1]

        for j in range(len(l)):
            ax.add_patch(
                patches.Rectangle(
                    (l[j][0], l[j][1]),
                    l[j][2],
                    l[j][3],
                    linewidth=2,
                    edgecolor="w",
                    facecolor="none",
                )
            )

    