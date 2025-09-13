import torch
import torch.multiprocessing as mp
from train import train_model
import os


def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("没有可用的GPU，无法进行DDP训练。请运行 train.py 进行单设备训练。")
        return
    elif world_size == 1:
        print("只有一个可用的GPU。建议直接运行 train.py 进行单卡训练。")

    print(f"发现 {world_size} 个GPU，将启动DDP训练...")
    mp.spawn(train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
