import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid

class VAE(nn.Module):

    def __init__(self, 
                 input_dim: int = 784, 
                 latent_dim: int = 200, 
                 hidden_dims: list = None, 
                 **kwargs) -> None:
        
        super(VAE, self).__init__()

        # initialize latent dimensions
        self.latent_dim = latent_dim

        # initialize hidden dimensions
        if hidden_dims is None:
            if latent_dim > 256:
                raise ValueError("Latent Dim > 256!")
            hidden_dims = [input_dim, 1024, 512, 256]

        decoder_hidden_dims = hidden_dims.copy()

        # Encoder Module Linear -> BatchNorm1d -> Sigmoid
        encoder_modules = []
        for i in range(0, len(hidden_dims) - 1):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Sigmoid()
                )
            )

        self.encoder = nn.Sequential(*encoder_modules)

        # Mean & Variance Networks
        self.mean_nn = nn.Linear(hidden_dims[-1], latent_dim)
        self.var_nn = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_hidden_dims.reverse()

        # first layer
        decoder_modules = [nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dims[0]),
            nn.BatchNorm1d(decoder_hidden_dims[0]),
            nn.Sigmoid()
        )]

        # consequent layers
        for i in range(len(decoder_hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(decoder_hidden_dims[i], decoder_hidden_dims[i + 1]),
                    nn.BatchNorm1d(decoder_hidden_dims[i + 1]),
                    nn.Sigmoid()
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

    # forward propagation
    def forward(self, x):

        x           = self.encoder(x)
        mean, var   = self.mean_nn(x), self.var_nn(x)
        std         = (0.5 * var).exp()
        epsilon     = torch.randn_like(std).to(device)
        z           = mean + std * epsilon
        z           = self.decoder(z)
        return z, mean, var

def calculateLoss(x, x_hat, mean, var):

    recon_loss  = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD         = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())  
    return recon_loss + KLD

# initialize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch version
print(f"Using Torch ({torch.__version__})")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# create sample generation directories
os.makedirs("vae_samples", exist_ok=True)

# convert raw data into tensor
transform_to_tensor = transforms.Compose([transforms.ToTensor()])

# dataset path
path = "./dataset"

# initialize datasets
train_dataset   = MNIST(path, transform=transform_to_tensor, download=True)
test_dataset    = MNIST(path, transform=transform_to_tensor, download=True)

# batch size
batch_size = 32

# initialize loaders
train_loader    = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader     = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

features, labels = next(iter(train_loader))

plt.imshow(features[3].squeeze())
plt.show()

# training parameters
epochs          = 30
input_dim       = 28 * 28
latent_dim      = 220
learning_rate   = 0.001

# initialize model
vae = VAE(input_dim, latent_dim).to(device)

# initialize optimizer
optimizer = Adam(vae.parameters(), lr=learning_rate)

# training
def train(): 
    for epoch in range(epochs):

        # loss computed for whole batch
        batch_loss = 0

        for i, (features, labels) in enumerate(train_loader):
            
            # send features to GPU (IF available) & flatten them into
            # shape (batch_size, 784)
            x = features.to(device)
            x = x.view(x.size(0), -1) 

            # zero gradients
            optimizer.zero_grad()
            # forward propagate
            x_hat, mean, var = vae.forward(x)
            # calculate loss
            loss = calculateLoss(x, x_hat, mean, var)
            # append loss
            batch_loss += loss.item()
            # calculate gradients
            loss.backward()
            # apply backpropagation
            optimizer.step()

        # save generated images
        with torch.no_grad():

            x_hat, _, _ = vae.forward(x)
            # reshape
            x_hat = x_hat.view(-1, 1, 28, 28)
            # save as png
            save_image(make_grid(x_hat.cpu(), nrow=8, normalize=True), f"vae_samples/vae_epoch_{epoch}.png")

        print(f"Epoch {epoch} Completed with Loss = {batch_loss:.4f}")

train()