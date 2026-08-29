import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import logging
import tqdm
from pathlib import Path
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader


logger = logging.getLogger(Path(__file__).name)


# region Distributed architecture
def weight_initialization(
    weight: torch.Tensor | DTensor,
    bias: torch.Tensor | DTensor | None = None,
) -> None:
    """
    Reset the coefficients of the weight matrix to the bounds of the Kaiming uniform
    initialization, and if a bias is provided, set it to zero.
    """
    match weight:
        case DTensor():
            # When distributed, initialize each shard manually to ensure same
            # statistical properties when not being distributed
            generator = torch.Generator().manual_seed(dist.get_rank())

            with torch.no_grad():
                local_weight = weight.to_local()

                # Initialize weights of the shard manually (nn.init.kaiming_uniform would
                # determine fan_in based on local shard size)
                if weight.dim() == 4:
                    # Weight from a convolutional layer
                    fan_in = weight.size(1) * weight.size(2) * weight.size(3)
                elif weight.dim() == 2:
                    # Weight from a fully connected layer.
                    fan_in = weight.size(1)
                else:
                    raise ValueError(f"Unsupported weight dimension: {weight.dim()}. Options "
                                     f"are 2 for fully connected layers, and 4 for conv2d layers")

                gain = nn.init.calculate_gain("tanh")
                std = gain / math.sqrt(fan_in)
                bound = math.sqrt(3.0) * std
                local_weight.uniform_(-bound, bound, generator=generator)

        case torch.Tensor():
            # Use standard initialization when conv layer weights are a tensor
            nn.init.kaiming_uniform_(weight, nonlinearity="tanh")

    match bias:
        case DTensor():
            # Start with a bias of zero per feature map
            local_bias = bias.to_local()
            nn.init.zeros_(local_bias)

        case torch.Tensor():
            nn.init.zeros_(bias)


class DistributedConv2d(nn.Conv2d):
    """
    A conv 2d layer whose weights and bias are sharded on the output channel dimension (dim 0).
    The convolution operation uses the local tensors of the input, weights and bias to perform the
    convolution, and does not return a DTensor. Instead, it returns a normal tensor for subsequent
    non-linear activation functions and pooling operations that only need to work with the local tensor.
    """
    def __init__(self, device_mesh: DeviceMesh, *args, **kwargs):
        super(DistributedConv2d, self).__init__(*args, **kwargs)
        self.device_mesh = device_mesh
        self.weight = nn.Parameter(distribute_tensor(self.weight, self.device_mesh, [Shard(0)]))
        if self.bias is not None:
            self.bias = nn.Parameter(distribute_tensor(self.bias, self.device_mesh, [Shard(0)]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d convolution on a replicated DTensor and passes the local tensor to the next layer.
        """
        if isinstance(input, DTensor) and isinstance(self.weight, DTensor):
            # First get the local tensors
            x = input.to_local()
            w = self.weight.to_local()
            b = self.bias.to_local() if self.bias is not None and isinstance(self.bias, DTensor) else None

            # Run the conv 2d function
            conv_params = (dict(stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))
            out = F.conv2d(input=x, weight=w, bias=b, **conv_params)

        else:
            # In case for normal tensors
            out = super().forward(input)

        return out


class DistributedAvgPool2d(nn.AvgPool2d):
    """
    An average pooling layer that expects local tensors from the individual shards, performs the
    pooling operation and returns a DTensor with the correct shard placement on the device mesh.
    """
    def __init__(self, *args, device_mesh: DeviceMesh, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_mesh = device_mesh

    def forward(self, input: torch.Tensor) -> DTensor:
        x = super().forward(input)
        return DTensor.from_local(x, self.device_mesh, [Shard(1)])


class DistributedLeNet5(nn.Module):
    """
    A tensor parallelized implementation of the LeNet architecture.
    """
    def __init__(self, device_mesh: DeviceMesh, n_classes: int = 10, in_channels: int = 1):
        super(DistributedLeNet5, self).__init__()
        self.device_mesh = device_mesh

        # Shard the weight and bias tensors across their output channels
        self.c1 = DistributedConv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            device_mesh=self.device_mesh
        )

        self.c2 = DistributedConv2d(
            in_channels=6,
            out_channels=18,
            kernel_size=5,
            device_mesh=self.device_mesh
        )

        self.c3 = DistributedConv2d(
            in_channels=18,
            out_channels=120,
            kernel_size=5,
            device_mesh=self.device_mesh
        )

        # Define pooling layers
        self.s1 = DistributedAvgPool2d(kernel_size=2, device_mesh=self.device_mesh)
        self.s2 = DistributedAvgPool2d(kernel_size=2, device_mesh=self.device_mesh)

        # Define the fully connected layers fc1 and fc2. Both layers will be parallelized
        # using (fc1) column-wise and (fc2) row-wise matrix multiplication, similar to the
        # megatron-lm paper. RowwiseParallel will implicitly conduct all_reduce summing step
        # across all nodes.
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=n_classes)

        self.fc1 = parallelize_module(self.fc1, device_mesh, ColwiseParallel())
        self.fc2 = parallelize_module(self.fc2, device_mesh, RowwiseParallel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Process each feature map for c1 block in parallel (conv -> tanh -> pooling)
        x = distribute_tensor(x, self.device_mesh, [Replicate()])
        x = self.s1(F.tanh(self.c1(x))).full_tensor()

        # Process each feature map for c2 block in parallel (conv -> tanh -> pooling)
        x = DTensor.from_local(x, self.device_mesh, [Replicate()])
        x = self.s2(F.tanh(self.c2(x))).full_tensor()

        # Process each feature map for c3 block in parallel (conv -> tanh -> flattening)
        x = DTensor.from_local(x, self.device_mesh, [Replicate()])
        x = F.tanh(self.c3(x))
        x = DTensor.from_local(x, self.device_mesh, [Shard(1)]).full_tensor()
        x = x.view(batch_size, -1)

        # Distribute the intermediate results from the feature extraction part across all processes
        x = DTensor.from_local(x, self.device_mesh, [Replicate()])
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def reset_parameters(self) -> None:
        """
        Use Kaiming uniform distribution to initialize the distributed weights and biases.
        """
        for c in [self.c1, self.c2, self.c3]:
            weight_initialization(c.weight, c.bias)

        # Also initialize the fully connected layers with the same method.
        weight_initialization(self.fc1.weight, self.fc1.bias)
        weight_initialization(self.fc2.weight, self.fc2.bias)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        nb_epochs: int = 100,
        device: str = "cpu",
        primary_node: bool = False
    ) -> None:
        """
        Training procedure for the distributed LeNet architecture.
        """
        for i in range(1, nb_epochs + 1):
            train_running_loss = 0
            valid_running_loss = 0
            sample_counter = 0

            # Only print progress bar on the primary rank
            with tqdm.tqdm(total=len(train_loader) + len(valid_loader), disable=not primary_node) as pbar:
                # Perform training iteration (1 Epoch)
                self.train()
                for x_train, y_train in train_loader:
                    # Move training data to target device
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)

                    # Clear gradient fields
                    optimizer.zero_grad()

                    # Forward pass
                    logits = self(x_train)
                    # logger.info(f"{logits.size()}, {y_train.size()}")
                    loss = loss_function(logits, y_train)
                    train_running_loss += loss.item() * x_train.size(0)
                    sample_counter += x_train.size(0)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Update description at each update step
                    pbar.set_description(f"Epoch {i}: Train loss: {train_running_loss / sample_counter:.04f} "
                                         f"-- Validation loss: ......")
                    pbar.update(1)

                # Evaluate on validation data (After 1 epoch)
                train_running_loss = train_running_loss / sample_counter
                self.eval()
                sample_counter = 0
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        # Move validation data to target device
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)

                        # Forward pass and record loss
                        logits = self(x_val)
                        loss = loss_function(logits, y_val)
                        valid_running_loss += loss.item() * x_val.size(0)
                        sample_counter += x_val.size(0)

                        # Update description at each validation step
                        pbar.set_description(f"Epoch {i}: Train loss: {train_running_loss:.04f} "
                                             f"-- Validation loss: {valid_running_loss / sample_counter:.04f}")
                        pbar.update(1)

    def save_model(self, checkpoint_dir: Path, rank: int, epoch: int):
        """
        Save the partial model weights in one rank to file.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        state_dict = {}
        for name, param in self.state_dict().items():
            # First convert distributed tensors to local tensors
            if isinstance(param, DTensor):
                state_dict[name] = param.to_local()

            else:
                state_dict[name] = param

        # Store the partial model of current rank in its own file
        torch.save(state_dict, checkpoint_dir / f"rank={rank:02d}_epoch={epoch:03d}.pt")

    def load_model(self, checkpoint_dir: Path, rank: int, epoch: int = -1) -> tuple[int, int]:
        """
        For the given rank, load all respective shards of the model weights.
        """
        if epoch < 0:
            # Determine the last epoch
            last_epoch = epoch
            for partial_checkpoint in sorted(checkpoint_dir.glob("*.pt")):
                if (match := re.search(r"rank=(\d+)_epoch=(\d+)\.pt", partial_checkpoint.name)) is not None:
                    e = int(match.group(2))
                    if e > last_epoch:
                        last_epoch = e

                else:
                    logger.warning(f"Checkpoint file {partial_checkpoint.name} does not seem to match filename format.")

            epoch = last_epoch

        # Load the data for the current rank for an epoch
        checkpoint_name = checkpoint_dir / f"rank={rank:02d}_epoch={epoch:03d}.pt"
        if not checkpoint_name.exists():
            logger.error(f"No checkpoint file could be found for rank={rank:02d} and epoch {epoch:03d}.")
            exit(1)

        state_dict = torch.load(checkpoint_name)
        for name, param in self.named_parameters():
            if name in state_dict:
                weight_shard = state_dict[name]
                # if isinstance(param, DTensor):
                weight_shard = DTensor.from_local(weight_shard, param.device_mesh, param.placements)

                # Copy the data from the partial weights into the distributed tensor
                param.data.copy_(weight_shard)

        return rank, epoch
# endregion