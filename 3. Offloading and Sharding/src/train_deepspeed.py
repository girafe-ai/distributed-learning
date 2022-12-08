import argparse

import torch
from tqdm.auto import tqdm

import subprocess
from time import sleep

try:
    import deepspeed
except ImportError:
    print("I need to install deepspeed! Use `pip install deepspeed` command")

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler

from model import Net


def get_dataloader():
    dataset = MNIST('./mnist', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=16)


def train(args: argparse.Namespace):
    device = f"cuda:{torch.distributed.get_rank()}"
    local_rank = torch.distributed.get_rank()
    model = Net()
    
    print(f"Run {local_rank}")
    
    loader = get_dataloader()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
    )

    steps = 0
    epoch_loss = 0
    loader_range = tqdm(loader) if args.local_rank == 0 else loader
    for data, target in loader_range:
        model_engine.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model_engine(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        model_engine.backward()
        model_engine.step()
        steps += 1
        
        if (steps % 100 == 0) and (local_rank == 0):
            subprocess.run("nvidia-smi")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", default=2)
    parser.add_argument("--deepspeed_config", default="./deepspeed_config_stage2.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    deepspeed.init_distributed()
    train(args)
