"""
Distributed Stochastic Gradient Descent using PyTorch

This implementation demonstrates distributed training with:
1. Data parallelism across multiple processes
2. Gradient synchronization using all-reduce
3. DistributedDataParallel (DDP) wrapper
4. Manual gradient synchronization for educational purposes
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.multiprocessing as mp


# Simple dataset for demonstration
class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 10000, input_dim: int = 100):
        self.X = torch.randn(num_samples, input_dim)
        # Generate labels from a linear model with noise
        true_weights = torch.randn(input_dim, 1)
        self.y = (self.X @ true_weights + 0.1 * torch.randn(num_samples, 1)).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze()


def setup(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def average_gradients(model: nn.Module):
    """
    Manually average gradients across all processes.
    This is what DDP does automatically, shown here for educational purposes.
    """
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def train_with_manual_sync(
    rank: int,
    world_size: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.01
):
    """
    Training loop with manual gradient synchronization.
    Demonstrates the core mechanics of distributed SGD.
    """
    setup(rank, world_size)

    # Create model and move to appropriate device
    model = SimpleNet()

    # Synchronize initial model parameters across all processes
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Create dataset and distributed sampler
    dataset = SyntheticDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        epoch_loss = 0.0

        for batch_idx, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Synchronize gradients across all processes
            average_gradients(model)

            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

        # Only print from rank 0
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"[Manual Sync] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    cleanup()


def train_with_ddp(
    rank: int,
    world_size: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.01
):
    """
    Training loop using PyTorch's DistributedDataParallel.
    This is the recommended approach for production use.
    """
    setup(rank, world_size)

    # Create model and wrap with DDP
    model = SimpleNet()
    ddp_model = DDP(model)

    # Create dataset and distributed sampler
    dataset = SyntheticDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            outputs = ddp_model(X)
            loss = criterion(outputs, y)

            # Backward pass - DDP handles gradient sync automatically
            loss.backward()

            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"[DDP] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    cleanup()


class DistributedSGD:
    """
    A reusable class for distributed SGD training.
    Supports both manual gradient sync and DDP.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        use_ddp: bool = True
    ):
        self.base_model = model
        self.dataset = dataset
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_ddp = use_ddp

    def _worker(
        self,
        rank: int,
        world_size: int,
        epochs: int,
        batch_size: int,
        results_queue
    ):
        """Worker function for each distributed process."""
        setup(rank, world_size)

        # Clone model for this process
        model = type(self.base_model)(
            *self._get_model_args()
        ) if hasattr(self, '_get_model_args') else SimpleNet()

        # Load state dict from base model
        model.load_state_dict(self.base_model.state_dict())

        if self.use_ddp:
            model = DDP(model)
        else:
            # Sync initial parameters
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

        sampler = DistributedSampler(
            self.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        history = []
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_loss = 0.0

            for X, y in dataloader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()

                if not self.use_ddp:
                    average_gradients(
                        model.module if hasattr(model, 'module') else model
                    )

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history.append(avg_loss)

            if rank == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Send results back from rank 0
        if rank == 0:
            final_model = model.module if hasattr(model, 'module') else model
            results_queue.put({
                'state_dict': final_model.state_dict(),
                'history': history
            })

        cleanup()

    def fit(self, epochs: int = 10, batch_size: int = 32, world_size: int = 2):
        """
        Train the model using distributed SGD.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size per process
            world_size: Number of distributed processes

        Returns:
            Training history (loss per epoch)
        """
        results_queue = mp.Queue()

        mp.spawn(
            self._worker,
            args=(world_size, epochs, batch_size, results_queue),
            nprocs=world_size,
            join=True
        )

        # Get results from rank 0
        results = results_queue.get()
        self.base_model.load_state_dict(results['state_dict'])
        return results['history']


def run_manual_sync(world_size: int = 2):
    """Run training with manual gradient synchronization."""
    print("\n" + "=" * 50)
    print("Training with Manual Gradient Synchronization")
    print("=" * 50)
    mp.spawn(
        train_with_manual_sync,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def run_ddp(world_size: int = 2):
    """Run training with DistributedDataParallel."""
    print("\n" + "=" * 50)
    print("Training with DistributedDataParallel (DDP)")
    print("=" * 50)
    mp.spawn(
        train_with_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Number of processes (workers)
    WORLD_SIZE = 2

    print("Distributed Stochastic Gradient Descent Demo")
    print(f"Number of workers: {WORLD_SIZE}")

    # Run both methods
    run_manual_sync(WORLD_SIZE)
    run_ddp(WORLD_SIZE)

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
