import torch
import subprocess
from tqdm.auto import tqdm

from model import Net
from data import get_dataloader


def train():
    loader = get_dataloader()
    model = Net()
    device = torch.device('cuda:0')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    steps = 0
    epoch_loss = 0
    for data, target in tqdm(loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        steps += 1
        
        if (steps % 100 == 0):
            subprocess.run("nvidia-smi")

if __name__ == "__main__":
    train()
