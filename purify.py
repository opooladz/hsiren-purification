import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torchvision.transforms as T
import os

# Import CIFAR10 dataset
from torchvision.datasets import CIFAR10




class Sine(nn.Module):
    def forward(self, x):
        # The factor of 30 is used as in your initial definition.
        return torch.sin(30 * x)

class HSine(nn.Module):
    def forward(self, x):
        return torch.sin(torch.sinh(2 * x))

def positional_encoding(coords, L=10):
    # This positional encoding concatenates sine and cosine transforms
    # for frequencies 1, 2, ..., L.
    encoded = []
    for i in range(1, L + 1):
        encoded.append(torch.sin(coords * i))
        encoded.append(torch.cos(coords * i))
    return torch.cat(encoded, dim=-1)


class SIREN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        input_dim = 2 * 10 * 2  # 2D coordinates * L=10 * (sin + cos)

        # Encoder: reduce to a latent bottleneck of dimension 1.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # encoder[0]
            HSine(),                           # encoder[1]
            nn.Linear(hidden_dim, 8),          # encoder[2]
            Sine()                            # encoder[3]
        )

        # Decoder: reconstruct from the latent code.
        self.decoder = nn.Sequential(
            nn.Linear(8, hidden_dim),         # decoder[0]
            nn.SiLU(),                           # decoder[1]
            nn.Linear(hidden_dim, 3)           # decoder[2]
        )

    def forward(self, coords):
        x = positional_encoding(coords).to(coords.dtype)
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent


def create_coords(device):
    # Create a grid of coordinates.
    x = torch.linspace(0, 32, 32)
    y = torch.linspace(0, 32, 32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    return coords


def train_single_image(img, additional_tensor, device):
    # Add perturbation to the image.
    img = torch.clamp(img + additional_tensor, -1, 1)

    coords = create_coords(device)
    pixels = img.permute(1, 2, 0).reshape(-1, 3)
    epochs = 1000
    siren = SIREN().to(device)
    # Optionally compile the model if using torch.compile (PyTorch 2.0+)
    siren = torch.compile(siren)

    # Select parameters for the Nuon optimizer.
    # Here, we choose the weight of the encoder's second linear layer and the decoder's first linear layer.
    nuon_params = [siren.encoder[2].weight, siren.decoder[0].weight]
    adamw_params = [
        p for n, p in siren.named_parameters()
        if n not in ('encoder.2.weight', 'decoder.0.weight')
    ]

    optimizer = Nuon(
        nuon_params,
        lr=0.00035,
        lr_param=1.0,
        momentum=0.95,
        adamw_params=adamw_params,
        adamw_lr=3e-4,
        adamw_betas=(0.90, 0.95),
        adamw_wd=0,
        whitening_prob=0.1
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: 1 - it / epochs)

    losses = []
    for i in range(epochs):
        out, latent = siren(coords)  # Unpack the tuple returned by forward
        loss = nn.MSELoss()(out, pixels)  # Use only the output for the loss calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if i % 100 == 0:
            print(f"Loss at iteration {i}: {loss.item():.6f}")

    return siren, optimizer, losses


def visualize(img, siren, device):
    coords = create_coords(device)
    with torch.no_grad():
        out, _ = siren(coords)
        pred_img = (out.reshape(32, 32, 3).cpu() + 1) / 2
        pred_img = torch.clamp(pred_img, 0, 1)

    original_img = (img.permute(1, 2, 0).cpu() + 1) / 2

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img.numpy())
    plt.title('Original Image with Perturbation')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_img.numpy())
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def train_multiple_images():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the perturbation tensor from 0.pth.
    # Assumes that 0.pth contains a tuple with (image_label_list, id_tensor, image_tensor)
    # and that image_tensor[0] is the desired perturbation.
    data = torch.load('/content/0.pth', map_location='cpu')
    image_label_list, id_tensor, image_tensor = data
    additional_tensor = torch.tensor(image_tensor[0]).to(device)

    # Define transforms for CIFAR10 images: normalize images to [-1, 1].
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: (x * 2) - 1)
    ])

    # Load CIFAR10 training dataset.
    cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # For demonstration, use the first 5 images from CIFAR10.
    for i in range(5):
        img, label = cifar_dataset[i]
        img = img.to(device)
        # Train on the CIFAR10 image while adding the perturbation from 0.pth.
        siren, optimizer, losses = train_single_image(img, additional_tensor, device)
        img_perturbed = torch.clamp(img + additional_tensor, -1, 1)
        visualize(img_perturbed, siren, device)


if __name__ == "__main__":
    train_multiple_images()
