import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from model import SegmentationModel
from config import Config
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')


def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    if len(s) % 2 != 0:
        return np.zeros(shape, dtype=np.uint8)

    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        lo = max(0, lo)
        hi = min(shape[0] * shape[1], hi)
        if lo < hi:
            img[lo:hi] = 1

    return img.reshape(shape, order='F')


class DefectDataset(Dataset):
    def __init__(self, csv_file, image_dir, image_size=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.image_size = image_size or Config.SEVERSTAL_IMAGE_SIZE
        self.transform = transform
        self.image_groups = self.df.groupby('ImageId')
        self.image_ids = list(self.image_groups.groups.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size[::-1]

        resize_size = (self.image_size[1], self.image_size[0])
        image = image.resize(resize_size)
        image = np.array(image) / 255.0

        mask = np.zeros((4, *self.image_size), dtype=np.float32)
        group = self.image_groups.get_group(image_id)

        for _, row in group.iterrows():
            class_id = int(row['ClassId']) - 1
            if not pd.isna(row['EncodedPixels']):
                decoded_mask = rle_decode(row['EncodedPixels'], original_size)
                decoded_mask = cv2.resize(decoded_mask, resize_size, interpolation=cv2.INTER_NEAREST)
                mask[class_id] = decoded_mask

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)
        mask = torch.FloatTensor(mask)
        return image, mask


class KSDD2DefectDataset(Dataset):
    def __init__(self, data_dir, image_size=None, transform=None):
        self.data_dir = data_dir
        self.image_size = image_size or Config.KSDD2_IMAGE_SIZE
        self.transform = transform

        all_files = os.listdir(data_dir)
        print(f"数据目录中总共有 {len(all_files)} 个文件")

        self.image_files = [f for f in all_files if f.endswith('.png') and not f.endswith('_GT.png')]
        print(f"找到 {len(self.image_files)} 个候选图片文件")

        valid_images = []
        missing_gt_files = []

        for img_file in self.image_files:
            gt_file = img_file.replace('.png', '_GT.png')
            gt_path = os.path.join(data_dir, gt_file)
            if os.path.exists(gt_path):
                valid_images.append(img_file)
            else:
                missing_gt_files.append(gt_file)

        self.image_files = valid_images
        print(f"找到 {len(self.image_files)} 对有效的图片-标签对")

        if len(missing_gt_files) > 0:
            print(f"缺失的GT文件数量: {len(missing_gt_files)}")

        if len(self.image_files) == 0:
            raise ValueError("没有找到有效的图片-标签对")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        gt_file = img_file.replace('.png', '_GT.png')

        image_path = os.path.join(self.data_dir, img_file)
        image = Image.open(image_path).convert('RGB')

        mask_path = os.path.join(self.data_dir, gt_file)
        mask = Image.open(mask_path).convert('L')

        original_size = image.size
        target_size = (self.image_size[1], self.image_size[0])

        pad_w = target_size[0] - original_size[0]
        pad_h = target_size[1] - original_size[1]

        from PIL import ImageOps
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        image = np.array(image) / 255.0
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = mask[np.newaxis, ...]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)

        mask = torch.FloatTensor(mask)
        return image, mask


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def plot_loss_curves(train_losses, test_losses, save_path='loss_curves.png'):
    """绘制训练和测试loss曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss曲线已保存到: {save_path}")


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 从配置文件获取参数
    dataset_config = Config.get_dataset_config()

    if Config.DATASET_TYPE == "severstal":
        dataset = DefectDataset(
            dataset_config['csv_file'],
            dataset_config['image_dir'],
            dataset_config['image_size']
        )
    elif Config.DATASET_TYPE == "KSDD2":
        dataset = KSDD2DefectDataset(
            dataset_config['data_dir'],
            dataset_config['image_size']
        )
    else:
        raise ValueError(f"不支持的数据集类型: {Config.DATASET_TYPE}")

    total_size = len(dataset)
    train_size = int(Config.TRAIN_RATIO * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"train size={train_size}, test size={test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )

    model = SegmentationModel(
        model_type=Config.MODEL_TYPE,
        num_classes=dataset_config['num_classes'],
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    test_losses = []

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{Config.NUM_EPOCHS} ===")

        model.train()
        total_loss = 0

        train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch + 1}',
                          leave=False, dynamic_ncols=True)

        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            bce_loss = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'BCE': f'{bce_loss.item():.4f}',
                'Dice': f'{d_loss.item():.4f}'
            })

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0

        test_pbar = tqdm(test_loader, desc=f'测试 Epoch {epoch + 1}',
                         leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for images, masks in test_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                bce_loss = criterion(outputs, masks)
                d_loss = dice_loss(outputs, masks)
                loss = bce_loss + d_loss
                test_loss += loss.item()

                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'BCE': f'{bce_loss.item():.4f}',
                    'Dice': f'{d_loss.item():.4f}'
                })

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        scheduler.step(avg_train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{Config.NUM_EPOCHS}:')
        print(f'  训练Loss: {avg_train_loss:.4f}')
        print(f'  测试Loss: {avg_test_loss:.4f}')
        print(f'  学习率: {current_lr:.6f}')

        if (epoch + 1) % 10 == 0:
            model_path = Config.get_model_path()
            torch.save(model.state_dict(), model_path)
            print(f'模型已保存到: {model_path}')

            plot_loss_curves(train_losses, test_losses,
                             f'loss_curves/{Config.MODEL_TYPE}_{Config.ENCODER_NAME}_{Config.DATASET_TYPE}/loss_curves_epoch_{epoch + 1}.png')

    final_model_path = Config.get_model_path()
    torch.save(model.state_dict(), final_model_path)

    plot_loss_curves(train_losses, test_losses,
                     f'loss_curves/{Config.MODEL_TYPE}_{Config.ENCODER_NAME}_{Config.DATASET_TYPE}/final_loss_curves.png')

    loss_data = {
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'test_loss': test_losses
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(f'loss_curves/{Config.MODEL_TYPE}_{Config.ENCODER_NAME}_{Config.DATASET_TYPE}/loss_history.csv',
                   index=False)


if __name__ == "__main__":
    train_model()