# Offloading and Sharding

Привет! Тут находятся примеры AMP и deepspeed.

1. train_amp.py - пример использования AMP для ускорения обучения модели.
2. train_deepspeed.py - пример использования deepspeed для ускорения обучения модели. Для него есть конфиг файлы, deepspeed_config_stage2.json и deepspeed_config_cpu_offload.json. Первый конфиг использует только распределенные вычисления, второй конфиг использует распределенные вычисления и offloading на CPU.
