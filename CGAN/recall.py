import os
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch
import torchvision
import numpy as np

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(batch_size, latent_size)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels)
    g_input = torch.cat((embedding(labels), z), 1).to(device)
    gen_imgs = G(g_input)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
batch_size = 100
num_epochs = 200
image_size = 784
n_class = 10
sample_dir = 'sample'
model_dir = 'model'

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Mnist dataset
mnist = torchvision.datasets.MNIST(
    root='../data/',
    transform=transform,
    train=True,
    download=True
)

# Data loader
data_loader = torch.utils.data.DataLoader(
    dataset=mnist,
    batch_size=batch_size,
    shuffle=True
)

# Generator
G = nn.Sequential(
    nn.Linear(n_class + latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)

# Discriminator
D = nn.Sequential(
    nn.Linear(n_class + image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)

G = G.to(device)
D = D.to(device)

# Criterion and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

#embedding
embedding = nn.Embedding(n_class, n_class)

# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images = images.view(batch_size, -1).to(device)
        labels = embedding(labels).to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        gen_labels = torch.randint(10, (batch_size, ))
        z = torch.randn(batch_size, latent_size)
        gen_input = torch.cat((embedding(gen_labels), z), 1).to(device)
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        d_input_real = torch.cat((labels, images), 1)

        gen_images = G(gen_input)
        d_input_fake = torch.cat((labels, gen_images), 1)

        output_real = D(d_input_real)
        d_loss_real = criterion(output_real, real_labels)

        output_fake = D(d_input_fake)
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # gen_labels = torch.randint(10, (batch_size,))
        z = torch.randn(batch_size, latent_size)
        gen_input = torch.cat((embedding(gen_labels), z), 1).to(device)
        gen_images = G(gen_input)
        d_input_fake = torch.cat((labels, gen_images), 1)

        output = D(d_input_real)
        g_loss = criterion(output, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [validity_real: %f] [validity_fake: %f]"
                % (epoch, num_epochs, i, len(data_loader), d_loss.item(), g_loss.item(), output_real.mean().item(), output_fake.mean().item())
            )

        batches_done = epoch * len(data_loader) + i
        if batches_done % 400 == 0:
            sample_image(n_row=10, batches_done=batches_done)












