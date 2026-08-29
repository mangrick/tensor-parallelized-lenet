# Tensor parallelism applied to LeNet
This project demonstrates tensor parallelism for deep neural networks using PyTorch's distributed library. 
LeNet-5 is used as a simple architecture to emphasize core concepts of parallelizing modules, working with shards of weights, and communication operations to distribute and gather tensors.
The project focuses on two different strategies: (1) Manual sharding in the feature extraction part of the architecture by applying tensor parallelism across the convolutional and pooling operations, and (2) the column-wise / row-wise matrix multiplication pattern from Megatron-LM to minimize communication between fully-connected layers.

## Obtaining the data
The MNIST dataset can be downloaded through torchvision using the following command (both training and test data will be downloaded).
Both train and evaluation scripts assume that the data has already been downloaded.
```bash
python -c "from torchvision import datasets; datasets.MNIST(root='mnist_data', train=True, download=True); datasets.MNIST(root='mnist_data', train=False, download=True)"
```

## Training the model
The following command shows how to start the training process with torchrun. 
The training script has a command line argument for the number of training epochs.
All checkpoint files for each rank will be stored in a *checkpoints* directory, where each checkpoint file contains the rank and the last epoch of that training run. 
```bash
OMP_NUM_THREADS=2 torchrun --nproc_per_node=3 train.py --n_epochs 3
```

## Evaluation on test data
The evaluation script requires that a model has already been trained for a given epoch, since the script will use those weights.
Model evaluation can be done using the command below. 
Keep in mind that the number of processes need to be the same as for the training script.
```bash
OMP_NUM_THREADS=2 torchrun --nproc_per_node=3 eval.py --epoch 3
```
