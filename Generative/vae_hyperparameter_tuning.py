from vae import VAE, Adam, DataLoader, train_dataset, calculateLoss, device
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import optuna
import matplotlib.pyplot as plt

def train_vae(config, checkpoint_dir=None):

    vae = VAE(input_dim=28*28, latent_dim=config["latent_dim"], hidden_dims=config["hidden_dims"]).to(device)
    optimizer = Adam(vae.parameters(), lr=config["learning_rate"])
    criterion = calculateLoss

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["epochs"]):
        batch_loss = 0
        vae.train()
        for i, (features, labels) in enumerate(train_loader):
            x = features.to(device)
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            x_hat, mean, var = vae(x)
            loss = criterion(x, x_hat, mean, var)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        tune.report()


config = {
    "latent_dim": tune.choice([100, 150, 200, 250]),  
    "learning_rate": tune.loguniform(1e-5, 1e-2),  
    "batch_size": tune.choice([16, 32, 64]),  
    "hidden_dims": tune.grid_search([[784, 1024, 512, 256], [784, 512, 256], [784, 1024, 512]]),  # Search for different hidden layer configurations
    "epochs": 30
}


scheduler = ASHAScheduler(
    metric="loss",  
    mode="min",     
    max_t=30,       
    grace_period=5,  
    reduction_factor=2
)

analysis = tune.run(
    train_vae,
    resources_per_trial={"cpu": 1, "gpu": 1},  
    config=config,
    scheduler=scheduler,
    num_samples=10,  
)

print("Best config found: ", analysis.best_config)
