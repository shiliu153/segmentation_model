import torch
import torch.multiprocessing as mp
from train import train_model
import os
import sys


def main():
    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法进行GPU训练")
        sys.exit(1)

    world_size = torch.cuda.device_count()

    if world_size == 0:
        print("错误: 没有可用的GPU，无法进行DDP训练")
        print("请检查CUDA安装和GPU驱动")
        sys.exit(1)
    elif world_size == 1:
        print("警告: 只有一个可用的GPU")
        print("建议直接运行 'python train.py' 进行单卡训练")
        print("继续使用DDP训练...")
    else:
        print(f"发现 {world_size} 个GPU，将启动DDP训练...")

    # 显示GPU信息
    for i in range(world_size):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    print("\n开始多进程分布式训练...")

    # 启动多进程训练
    try:
        mp.spawn(train_model,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
        print("\n分布式训练完成!")
    except Exception as e:
        print(f"分布式训练出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 设置多进程启动方法 - 非Windows系统使用fork，Windows使用spawn
    if sys.platform != 'win32':
        mp.set_start_method('fork', force=True)
    else:
        mp.set_start_method('spawn', force=True)
    main()