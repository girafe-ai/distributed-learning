import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }

    def train(self, train_data: Dict[str, np.array], batch_size: int = 1):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        
        for batch in train_dataloader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
