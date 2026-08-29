import argparse
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from pathlib import Path
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import DistributedLeNet5


logger = logging.getLogger(Path(__file__).name)


def calculate_model_size(model: nn.Module) -> tuple[int, int]:
    """
    Calculate the size of the model (without tensor parallelism) and the number
    of parameter each rank holds in memory.
    """
    nb_train_params = 0
    nb_rank_params = 0

    for p in model.parameters():
        if p.requires_grad:
            nb_train_params += p.numel()

            if isinstance(p, DTensor):
                local_tensor = p.to_local()
                nb_rank_params += local_tensor.numel()
            else:
                nb_rank_params += p.numel()

    return nb_train_params, nb_rank_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the distributed LeNet model.")
    parser.add_argument("--n_epochs", help="Number of epochs to train.", default=3)
    args = parser.parse_args()

    # Define number of epochs for training:
    n_epochs = int(args.n_epochs)

    # Obtain unique ID for each process across all machines
    rank = int(os.environ["RANK"])

    # Obtain total number of processes (across all machines)
    world_size = int(os.environ["WORLD_SIZE"])

    # Unique ID only for a specific machine (to select appropriate device number)
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the logger for the command line output
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(name)-15s] [RANK: {rank}] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S'
    )

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = "cpu"
    device_mesh = DeviceMesh(device, list(range(world_size)))

    # Initialize model on the meta device to avoid allocating memory for the internal parameters
    # before distributing them across different ranks
    with torch.device("meta"):
        model = DistributedLeNet5(device_mesh=device_mesh)

    # Allocate memory for the sharded weights on the target device, and initialize the weights
    model.to_empty(device=device)
    model.reset_parameters()

    # On first rank, log global and sharded memory weight sizes
    if rank == 0:
        total, param_per_rank = calculate_model_size(model)
        logger.info(f"Total number of parameters: {total}, "
                    f"number of parameters per rank: {param_per_rank} ({(param_per_rank / total) * 100:.02f}%)")

    # Define train and validation datasets for the mnist image files
    train_dataset = datasets.MNIST(
        root="mnist_data",
        train=True,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
        download=False
    )

    # Use 10% of the training data as a held-out validation set
    indices = torch.arange(len(train_dataset))
    split = math.floor(len(train_dataset) * 0.1)

    # Ensure that every rank gets the same batch of data in each iteration by using a constant seed
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(42))
    valid_sampler = SubsetRandomSampler(valid_indices, generator=torch.Generator().manual_seed(43))

    # Setup training and validation data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(dataset=train_dataset, batch_size=32, sampler=valid_sampler, num_workers=0)

    # Train the model on the training set and validate on held out unseen data
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, foreach=False)
    model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_function=criterion,
        optimizer=optimizer,
        nb_epochs=n_epochs,
        device=device,
        primary_node=rank == 0
    )

    model.save_model(checkpoint_dir=Path("./checkpoints"), rank=rank, epoch=n_epochs)
    dist.destroy_process_group()
