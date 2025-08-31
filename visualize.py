import torch
import matplotlib.pyplot as plt
from config import Config
import os

def visualize_detection_results(num_samples=10):
    save_dir = Config.get_visualization_dir()
    os.makedirs(save_dir, exist_ok=True)

    data = torch.load(Config.get_output_path(), map_location='cpu')
    pred_masks = data['pred_masks']
    binary_masks = data['binary_masks']
    true_masks = data['true_masks']
    dataset_type = data.get('dataset_type', 'severstal')
    num_classes = data.get('num_classes', 4)

    num_to_show = min(num_samples, len(pred_masks))

    if dataset_type == "KSDD2":
        for idx in range(num_to_show):
            pred_mask = pred_masks[idx]
            binary_mask = binary_masks[idx]
            true_mask = true_masks[idx]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f'Image {idx + 1} - KSDD2 Defect Detection', fontsize=16)

            axes[0].imshow(pred_mask[0], cmap='hot', vmin=0, vmax=1)
            axes[0].set_title('Prediction Score')
            axes[0].axis('off')

            axes[1].imshow(binary_mask[0], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Binary Mask')
            axes[1].axis('off')

            axes[2].imshow(true_mask[0], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'KSDD2_result_{idx + 1}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    else:
        class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

        for idx in range(num_to_show):
            pred_mask = pred_masks[idx]
            binary_mask = binary_masks[idx]
            true_mask = true_masks[idx]

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle(f'Image {idx + 1} - Severstal Detection Results', fontsize=16)

            for class_idx in range(4):
                axes[0, class_idx].imshow(pred_mask[class_idx], cmap='hot', vmin=0, vmax=1)
                axes[0, class_idx].set_title(f'{class_names[class_idx]}\nPrediction Score')
                axes[0, class_idx].axis('off')

                axes[1, class_idx].imshow(binary_mask[class_idx], cmap='gray', vmin=0, vmax=1)
                axes[1, class_idx].set_title(f'{class_names[class_idx]}\nBinary Mask')
                axes[1, class_idx].axis('off')

                axes[2, class_idx].imshow(true_mask[class_idx], cmap='gray', vmin=0, vmax=1)
                axes[2, class_idx].set_title(f'{class_names[class_idx]}\nGround Truth')
                axes[2, class_idx].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'severstal_result_{idx + 1}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    print(f"可视化结果已保存到 {save_dir}")
    print(f"数据集类型: {dataset_type}, 类别数: {num_classes}")

if __name__ == "__main__":
    visualize_detection_results(num_samples=20)