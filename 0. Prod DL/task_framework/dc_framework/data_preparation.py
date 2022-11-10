import torch


class Dataset:
    def __init__(self, data, dtype=torch.float32):
        self._data = {
            "feature": torch.from_numpy(data["feature"]).to(dtype),
            "target": torch.from_numpy(data["target"]).to(dtype),
        }

        if self._data["target"].dim() == 1:
            self._data["target"] = self._data["target"].unsqueeze(1)

    def __getitem__(self, index):
        return self._data["feature"][index], self._data["target"][index]

    def __len__(self):
        return self._data["feature"].size(0)

    def get_dataloader(self, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.default_collate_fn
        )
        return train_dataloader

    def default_collate_fn(self, batch):
        features = []
        targets = []

        for sample in batch:
            features.append(sample[0])
            targets.append(sample[1])

        if len(features) > 1:
            return torch.concat(features, 0), torch.concat(targets, 0)
        else:
            return features[0].unsqueeze(0), targets[0].unsqueeze(0)

