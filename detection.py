import torch
from torch.utils.data import DataLoader, Subset
from model import SegmentationModel
from train import DefectDataset, KSDD2DefectDataset
from config import Config
from tqdm import tqdm
import os


def get_detection_tensor_dataset():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    test_size = int(Config.TEST_RATIO * total_size)
    test_indices = list(range(total_size - test_size, total_size))
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SegmentationModel(
        model_type=Config.MODEL_TYPE,
        num_classes=dataset_config['num_classes'],
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS
    ).to(device)

    model.load_state_dict(torch.load(Config.get_model_path(), map_location=device))
    model.eval()

    pred_masks_list = []
    binary_masks_list = []
    true_masks_list = []

    for image, mask in tqdm(test_loader):
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
            score = torch.sigmoid(output).cpu().squeeze(0)
            binary = (score > Config.DETECTION_THRESHOLD).float()
            true_mask = mask.squeeze(0)

        pred_masks_list.append(score)
        binary_masks_list.append(binary)
        true_masks_list.append(true_mask)

    torch.save({
        'pred_masks': pred_masks_list,
        'binary_masks': binary_masks_list,
        'true_masks': true_masks_list,
        'image_count': len(pred_masks_list),
        'dataset_type': Config.DATASET_TYPE,
        'num_classes': dataset_config['num_classes']
    }, Config.get_output_path())

    print(f"保存了 {len(pred_masks_list)} 张图片的检测结果到 {Config.get_output_path()}")

    class_detection_count = [0] * dataset_config['num_classes']
    class_true_count = [0] * dataset_config['num_classes']

    for binary_masks, true_masks in zip(binary_masks_list, true_masks_list):
        for class_idx in range(dataset_config['num_classes']):
            if binary_masks[class_idx].sum() > 0:
                class_detection_count[class_idx] += 1
            if true_masks[class_idx].sum() > 0:
                class_true_count[class_idx] += 1

    print("各类别检测统计:")
    for class_idx in range(dataset_config['num_classes']):
        if Config.DATASET_TYPE == "KSDD2":
            print(f"  缺陷检测: 检测到 {class_detection_count[class_idx]} 张, 真实有 {class_true_count[class_idx]} 张")
        else:
            print(
                f"  Class {class_idx + 1}: 检测到 {class_detection_count[class_idx]} 张, 真实有 {class_true_count[class_idx]} 张")

    return len(pred_masks_list)


if __name__ == "__main__":
    result_count = get_detection_tensor_dataset()
    print(f"检测完成，共处理 {result_count} 张图片")