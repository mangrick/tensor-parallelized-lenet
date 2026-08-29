import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import logging
import tqdm
from pathlib import Path
from torch.distributed import DeviceMesh
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DistributedLeNet5


logger = logging.getLogger(Path(__file__).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for the distributed LeNet model.")
    parser.add_argument("--epochs", help="Which epoch to load model weights. Default is -1 for last epoch.", default=-1)
    args = parser.parse_args()

    # Which epoch to load
    epoch = int(args.epochs)

    # Get the unique ID for current process in the mesh
    rank = int(os.environ["RANK"])

    # Get the total number of processes (across all machines)
    world_size = int(os.environ["WORLD_SIZE"])

    # And the unique ID only for a specific machine
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the logger for the command line output
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(name)-15s] [RANK: {rank}] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S'
    )

    # Setup process group, device and the device mesh
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = "cpu"
    device_mesh = DeviceMesh(device, list(range(world_size)))

    # Similar to the training, we first only create the model on the meta device to not allocate memory for the weights
    with torch.device("meta"):
        model = DistributedLeNet5(device_mesh=device_mesh)

    # Allocate memory for the sharded weights on the target device, and load the weights from file
    model.to_empty(device=device)
    _, loaded_epoch = model.load_model(checkpoint_dir=Path("./checkpoints"), rank=rank, epoch=epoch)

    # Define the test dataset for the mnist image files
    test_dataset = datasets.MNIST(
        root="mnist_data",
        train=False,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
        download=False
    )

    # Set up the test dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Run the model in inference mode on the test data
    model.eval()
    correct = 0

    with torch.no_grad():
        for x_test, y_test in tqdm.tqdm(test_loader, desc="Evaluating", disable=rank != 0):
            # Move unseen data to target device
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # Forward pass and obtain predicted class
            logits = model(x_test)
            values = F.softmax(logits, dim=1).argmax(dim=1)
            correct += (values == y_test).sum().item()

    dist.destroy_process_group()

    if rank == 0:
        score = correct / len(test_dataset)
        logger.info(f"Final accuracy score on unseen data after {loaded_epoch} epochs: {score * 100:.2f}%")
