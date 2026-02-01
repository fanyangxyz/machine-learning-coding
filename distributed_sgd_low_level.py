"""
Distributed Stochastic Gradient Descent using Primitive Collective Operations

This implementation demonstrates distributed training using low-level primitives:
1. all_reduce - Sum/average gradients across all workers
2. all_gather - Gather tensors from all workers
3. reduce + broadcast - Reduce to one node, then broadcast
4. Ring all-reduce - Manual implementation of ring-based reduction
5. Parameter server pattern - Centralized gradient aggregation
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.multiprocessing as mp
from typing import List, Literal
from enum import Enum


class SyncStrategy(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_BROADCAST = "reduce_broadcast"
    RING_ALL_REDUCE = "ring_all_reduce"
    PARAMETER_SERVER = "parameter_server"


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 10000, input_dim: int = 100):
        self.X = torch.randn(num_samples, input_dim)
        true_weights = torch.randn(input_dim, 1)
        self.y = (self.X @ true_weights + 0.1 * torch.randn(num_samples, 1)).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


# =============================================================================
# Primitive Gradient Synchronization Strategies
# =============================================================================

def sync_gradients_all_reduce(model: nn.Module):
    """
    Strategy 1: ALL_REDUCE

    All workers collectively reduce gradients and all receive the result.
    This is the most common and efficient approach.

    Before: Worker 0 has g0, Worker 1 has g1, Worker 2 has g2
    After:  All workers have (g0 + g1 + g2) / 3

    Communication: O(N) data, O(log N) or O(1) steps depending on implementation
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            # Sum gradients across all processes
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            # Average the gradients
            param.grad.data /= world_size


def sync_gradients_all_gather(model: nn.Module):
    """
    Strategy 2: ALL_GATHER

    Each worker gathers all gradients, then computes the average locally.
    More memory intensive but useful when you need all individual gradients.

    Before: Worker 0 has g0, Worker 1 has g1
    After:  All workers have [g0, g1] and compute mean locally

    Communication: O(N * world_size) data
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data

            # Prepare tensor list to gather into
            gathered_grads = [torch.zeros_like(grad) for _ in range(world_size)]

            # Gather gradients from all workers
            dist.all_gather(gathered_grads, grad)

            # Average the gathered gradients locally
            param.grad.data = torch.stack(gathered_grads).mean(dim=0)


def sync_gradients_all_gather_into_tensor(model: nn.Module):
    """
    Strategy 2b: ALL_GATHER_INTO_TENSOR

    More efficient version using contiguous tensor output.
    Gathers all gradients into a single pre-allocated tensor.
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_flat = grad.view(-1)

            # Pre-allocate flat output tensor (world_size * grad_size)
            output_tensor = torch.zeros(
                world_size * grad_flat.numel(),
                dtype=grad.dtype,
                device=grad.device
            )

            # Gather into contiguous tensor
            dist.all_gather_into_tensor(output_tensor, grad_flat)

            # Reshape and average the gathered gradients
            gathered = output_tensor.view(world_size, -1)
            param.grad.data = gathered.mean(dim=0).view(grad.shape)


def sync_gradients_reduce_broadcast(model: nn.Module, root: int = 0):
    """
    Strategy 3: REDUCE + BROADCAST

    Two-phase approach:
    1. Reduce all gradients to a single root node
    2. Broadcast the averaged result back to all nodes

    Before: Worker 0 has g0, Worker 1 has g1
    Step 1: Worker 0 (root) receives g0 + g1
    Step 2: Worker 0 broadcasts averaged gradient to all

    Communication: O(N) reduce + O(N) broadcast
    """
    world_size = dist.get_world_size()

    for param in model.parameters():
        if param.grad is not None:
            # Phase 1: Reduce to root
            dist.reduce(param.grad.data, dst=root, op=dist.ReduceOp.SUM)

            # Root computes average
            if dist.get_rank() == root:
                param.grad.data /= world_size

            # Phase 2: Broadcast from root to all
            dist.broadcast(param.grad.data, src=root)


def sync_gradients_ring_all_reduce(model: nn.Module):
    """
    Strategy 4: RING ALL-REDUCE (Manual Implementation)

    Implements ring-based all-reduce in two phases:
    1. Scatter-reduce: Each node reduces a chunk and passes to next
    2. All-gather: Each node broadcasts its chunk around the ring

    This is how NCCL implements all-reduce for efficiency.

    For N workers and data size D:
    - Communication: 2 * (N-1) * D/N ≈ 2D (bandwidth optimal)
    - Steps: 2 * (N-1)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data.view(-1)  # Flatten
            n = grad.numel()

            # Pad to be divisible by world_size
            pad_size = (world_size - n % world_size) % world_size
            if pad_size > 0:
                grad = torch.cat([grad, torch.zeros(pad_size, device=grad.device)])

            chunk_size = grad.numel() // world_size

            # Split into chunks
            chunks = list(grad.chunk(world_size))

            # Phase 1: Scatter-reduce
            # Each iteration, send chunk[i] to next, receive from prev, accumulate
            for step in range(world_size - 1):
                send_idx = (rank - step) % world_size
                recv_idx = (rank - step - 1) % world_size

                send_chunk = chunks[send_idx].clone()
                recv_chunk = torch.zeros_like(chunks[recv_idx])

                # Send to next rank, receive from previous rank
                next_rank = (rank + 1) % world_size
                prev_rank = (rank - 1) % world_size

                send_req = dist.isend(send_chunk, dst=next_rank)
                dist.recv(recv_chunk, src=prev_rank)
                send_req.wait()

                # Accumulate received chunk
                chunks[recv_idx] += recv_chunk

            # Phase 2: All-gather
            # Each iteration, send reduced chunk to next, receive from prev
            for step in range(world_size - 1):
                send_idx = (rank - step + 1) % world_size
                recv_idx = (rank - step) % world_size

                send_chunk = chunks[send_idx].clone()
                recv_chunk = torch.zeros_like(chunks[recv_idx])

                next_rank = (rank + 1) % world_size
                prev_rank = (rank - 1) % world_size

                send_req = dist.isend(send_chunk, dst=next_rank)
                dist.recv(recv_chunk, src=prev_rank)
                send_req.wait()

                chunks[recv_idx] = recv_chunk

            # Reconstruct and average
            result = torch.cat(chunks)[:n]  # Remove padding
            param.grad.data = (result / world_size).view(param.grad.shape)


def sync_gradients_parameter_server(model: nn.Module, server_rank: int = 0):
    """
    Strategy 5: PARAMETER SERVER Pattern

    Simulates a parameter server architecture:
    1. Workers send gradients to server using isend
    2. Server gathers and averages gradients
    3. Server sends updated gradients back to workers

    This pattern is common in large-scale distributed training.

    Communication: O(N * world_size) for gather + O(N * world_size) for scatter
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data

            if rank == server_rank:
                # Server: Gather gradients from all workers
                all_grads = [torch.zeros_like(grad) for _ in range(world_size)]
                all_grads[server_rank] = grad.clone()

                # Receive from all other workers
                reqs = []
                for worker in range(world_size):
                    if worker != server_rank:
                        req = dist.irecv(all_grads[worker], src=worker)
                        reqs.append(req)

                for req in reqs:
                    req.wait()

                # Average gradients
                avg_grad = torch.stack(all_grads).mean(dim=0)

                # Send averaged gradient back to all workers
                for worker in range(world_size):
                    if worker != server_rank:
                        dist.send(avg_grad, dst=worker)

                param.grad.data = avg_grad
            else:
                # Worker: Send gradient to server
                dist.send(grad, dst=server_rank)

                # Receive averaged gradient from server
                dist.recv(param.grad.data, src=server_rank)


# =============================================================================
# Training Functions
# =============================================================================

def get_sync_function(strategy: SyncStrategy):
    """Get the synchronization function for a given strategy."""
    sync_functions = {
        SyncStrategy.ALL_REDUCE: sync_gradients_all_reduce,
        SyncStrategy.ALL_GATHER: sync_gradients_all_gather_into_tensor,
        SyncStrategy.REDUCE_BROADCAST: sync_gradients_reduce_broadcast,
        SyncStrategy.RING_ALL_REDUCE: sync_gradients_ring_all_reduce,
        SyncStrategy.PARAMETER_SERVER: sync_gradients_parameter_server,
    }
    return sync_functions[strategy]


def train_worker(
    rank: int,
    world_size: int,
    strategy: SyncStrategy,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.01
):
    """Training worker using specified synchronization strategy."""
    setup(rank, world_size)

    # Create model
    model = SimpleNet()

    # Broadcast initial parameters from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Create dataset with distributed sampler
    dataset = SyntheticDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Get sync function
    sync_fn = get_sync_function(strategy)

    # Loss function
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for X, y in dataloader:
            # Zero gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass
            loss.backward()

            # Synchronize gradients using the chosen strategy
            sync_fn(model)

            # Manual SGD update
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad.data

            epoch_loss += loss.item()

        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"[{strategy.value}] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    cleanup()


def run_strategy(strategy: SyncStrategy, world_size: int = 2):
    """Run training with a specific synchronization strategy."""
    print(f"\n{'=' * 60}")
    print(f"Training with {strategy.value.upper()} strategy")
    print(f"{'=' * 60}")

    mp.spawn(
        train_worker,
        args=(world_size, strategy),
        nprocs=world_size,
        join=True
    )


# =============================================================================
# Demonstration of Individual Primitives
# =============================================================================

def demo_primitives(rank: int, world_size: int):
    """Demonstrate individual collective operations."""
    setup(rank, world_size)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Demonstrating Primitive Collective Operations")
        print("=" * 60)

    dist.barrier()

    # 1. BROADCAST: Send tensor from one rank to all
    tensor = torch.tensor([rank * 10.0, rank * 10.0 + 1])
    if rank == 0:
        print(f"\n1. BROADCAST (src=0)")
        print(f"   Before: rank {rank} has {tensor.tolist()}")
    dist.barrier()

    dist.broadcast(tensor, src=0)
    print(f"   After:  rank {rank} has {tensor.tolist()}")
    dist.barrier()

    # 2. REDUCE: Sum tensors to one rank
    tensor = torch.tensor([rank + 1.0, rank + 1.0])
    if rank == 0:
        print(f"\n2. REDUCE (dst=0, op=SUM)")
    dist.barrier()
    print(f"   Before: rank {rank} has {tensor.tolist()}")
    dist.barrier()

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"   After:  rank 0 has {tensor.tolist()} (sum of all)")
    dist.barrier()

    # 3. ALL_REDUCE: Reduce and distribute to all
    tensor = torch.tensor([rank + 1.0, rank + 1.0])
    if rank == 0:
        print(f"\n3. ALL_REDUCE (op=SUM)")
    dist.barrier()
    print(f"   Before: rank {rank} has {tensor.tolist()}")
    dist.barrier()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"   After:  rank {rank} has {tensor.tolist()}")
    dist.barrier()

    # 4. ALL_GATHER: Gather tensors from all ranks to all ranks
    tensor = torch.tensor([rank * 100.0])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    if rank == 0:
        print(f"\n4. ALL_GATHER")
    dist.barrier()
    print(f"   Before: rank {rank} has {tensor.tolist()}")
    dist.barrier()

    dist.all_gather(gathered, tensor)
    print(f"   After:  rank {rank} has gathered {[g.item() for g in gathered]}")
    dist.barrier()

    # 5. ALL_GATHER_INTO_TENSOR: Efficient version with contiguous output
    tensor = torch.tensor([rank * 100.0, rank * 100.0 + 1])
    output = torch.zeros(world_size * 2)  # Flat tensor: world_size * tensor_size
    if rank == 0:
        print(f"\n5. ALL_GATHER_INTO_TENSOR")
    dist.barrier()
    print(f"   Before: rank {rank} has {tensor.tolist()}")
    dist.barrier()

    dist.all_gather_into_tensor(output, tensor)
    print(f"   After:  rank {rank} has gathered: {output.tolist()}")
    dist.barrier()

    # 6. SCATTER: Distribute different tensors to each rank
    if rank == 0:
        scatter_list = [torch.tensor([i * 5.0]) for i in range(world_size)]
        print(f"\n6. SCATTER (src=0)")
        print(f"   Scattering: {[s.tolist() for s in scatter_list]}")
    else:
        scatter_list = None
    dist.barrier()

    output = torch.zeros(1)
    dist.scatter(output, scatter_list if rank == 0 else None, src=0)
    print(f"   rank {rank} received: {output.tolist()}")
    dist.barrier()

    # 7. GATHER: Collect tensors from all ranks to one
    tensor = torch.tensor([rank * 7.0])
    if rank == 0:
        gather_list = [torch.zeros(1) for _ in range(world_size)]
        print(f"\n7. GATHER (dst=0)")
    else:
        gather_list = None
    dist.barrier()
    print(f"   rank {rank} sending: {tensor.tolist()}")
    dist.barrier()

    dist.gather(tensor, gather_list if rank == 0 else None, dst=0)
    if rank == 0:
        print(f"   rank 0 gathered: {[g.tolist() for g in gather_list]}")
    dist.barrier()

    # 8. REDUCE_SCATTER: Reduce and scatter in one operation
    input_tensor = torch.tensor([rank + 1.0] * world_size)
    output_tensor = torch.zeros(1)
    if rank == 0:
        print(f"\n8. REDUCE_SCATTER")
    dist.barrier()
    print(f"   Before: rank {rank} has {input_tensor.tolist()}")
    dist.barrier()

    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"   After:  rank {rank} has {output_tensor.tolist()}")
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)

    cleanup()


def run_demo(world_size: int = 2):
    """Run the primitives demonstration."""
    mp.spawn(
        demo_primitives,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    WORLD_SIZE = 2

    print("Distributed SGD with Primitive Collective Operations")
    print(f"Number of workers: {WORLD_SIZE}")

    # First, demonstrate the primitives
    run_demo(WORLD_SIZE)

    # Then run training with each strategy
    strategies = [
        SyncStrategy.ALL_REDUCE,
        SyncStrategy.ALL_GATHER,
        SyncStrategy.REDUCE_BROADCAST,
        SyncStrategy.RING_ALL_REDUCE,
        SyncStrategy.PARAMETER_SERVER,
    ]

    for strategy in strategies:
        run_strategy(strategy, WORLD_SIZE)

    print("\n" + "=" * 60)
    print("All strategies completed!")
    print("=" * 60)
