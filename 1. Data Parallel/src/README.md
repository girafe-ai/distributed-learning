# Distributed Data Parallel

Привет! Тут находятся примеры, которые показывают, как использовать распределенные вычисления в PyTorch.

Для запуска `train.py` можно использовать обычный python:
```
python train.py
```

Для запуска `train_distributed.py` можно использовать `torch.distributed.launch`:
```
python -m torch.distributed.launch train_distributed.py
```
