import torch
import segmentation_models_pytorch as smp
from config import Config
from train import train_model
import time
import logging
import os
import sys
import json
import traceback
from datetime import datetime
from contextlib import contextmanager


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def get_all_model_types():
    return ["UNet", "UNetPlusPlus", "DeepLabV3", "DeepLabV3Plus", "FPN", "PSPNet", "Linknet", "MAnet"]


def get_priority_encoders():
    return [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'densenet121', 'densenet169',
        'mobilenet_v2',
        'se_resnet50'
    ]


def get_all_encoders():
    return [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
        'resnext101_32x32d', 'resnext101_32x48d',

        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
        'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
        'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5',

        'densenet121', 'densenet169', 'densenet201', 'densenet161',

        'vgg11', 'vgg13', 'vgg16', 'vgg19',
        'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',

        'mobilenet_v2', 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100',
        'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100',

        'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008',
        'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064',
        'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320',

        'se_resnet50', 'se_resnet101', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d',

        'timm-skresnet18', 'timm-skresnet34', 'timm-skresnet50', 'timm-skresnet101',

        'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e',
        'timm-resnest200e', 'timm-resnest269e',

        'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s',
        'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s',

        'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',

        'xception',

        'inceptionv4', 'inceptionresnetv2',

        'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l',
    ]


def create_model(model_type, encoder_name, num_classes):
    model_dict = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "linknet": smp.Linknet,
        "manet": smp.MAnet
    }

    model_class = model_dict.get(model_type.lower())
    if not model_class:
        return None

    return model_class(
        encoder_name=encoder_name,
        encoder_weights=Config.ENCODER_WEIGHTS,
        classes=num_classes,
        activation=None
    )


def validate_combination(model_type, encoder_name, num_classes):
    try:
        model = create_model(model_type, encoder_name, num_classes)
        if model is None:
            return False

        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            _ = model(test_input)

        del model, test_input
        torch.cuda.empty_cache()
        return True

    except Exception:
        return False


@contextmanager
def training_context(model_type, encoder_name):
    original_model = Config.MODEL_TYPE
    original_encoder = Config.ENCODER_NAME

    try:
        Config.MODEL_TYPE = model_type
        Config.ENCODER_NAME = encoder_name
        yield
    finally:
        Config.MODEL_TYPE = original_model
        Config.ENCODER_NAME = original_encoder


def save_training_results(results, timestamp):
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results_file


def generate_summary_report(results, logger):
    logger.info("\n" + "=" * 80)
    logger.info("训练总结报告")
    logger.info("=" * 80)

    total_combinations = len(results['combinations'])
    successful = len(results['successful'])
    failed = len(results['failed'])

    logger.info(f"总组合数: {total_combinations}")
    logger.info(f"成功训练: {successful} ({successful / total_combinations * 100:.1f}%)")
    logger.info(f"失败训练: {failed} ({failed / total_combinations * 100:.1f}%)")
    logger.info(f"总训练时间: {results['total_time']:.2f}秒 ({results['total_time'] / 3600:.2f}小时)")

    if results['successful']:
        logger.info("\n最佳表现组合:")
        best_combo = min(results['successful'], key=lambda x: x['training_time'])
        logger.info(
            f"  最快训练: {best_combo['model_type']} + {best_combo['encoder']} ({best_combo['training_time']:.2f}秒)")

        slowest_combo = max(results['successful'], key=lambda x: x['training_time'])
        logger.info(
            f"  最慢训练: {slowest_combo['model_type']} + {slowest_combo['encoder']} ({slowest_combo['training_time']:.2f}秒)")

    if results['failed']:
        logger.info(f"\n失败原因统计:")
        error_counts = {}
        for failure in results['failed']:
            error_type = failure['error'].split(':')[0] if ':' in failure['error'] else failure['error']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            logger.info(f"  {error_type}: {count} 次")


def batch_train_all_combinations(mode='priority', resume_from=None):
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法进行GPU训练")
        return

    dataset_config = Config.get_dataset_config()
    num_classes = dataset_config['num_classes']

    model_types = get_all_model_types()
    encoders = get_priority_encoders() if mode == 'priority' else get_all_encoders()

    logger.info(f"批量训练模式: {mode}")
    logger.info(f"数据集类型: {Config.DATASET_TYPE}")
    logger.info(f"类别数量: {num_classes}")
    logger.info(f"模型类型数量: {len(model_types)}")
    logger.info(f"编码器数量: {len(encoders)}")

    combinations = [(model, encoder) for model in model_types for encoder in encoders]
    total_combinations = len(combinations)
    logger.info(f"总组合数: {total_combinations}")

    logger.info("验证可用组合...")
    valid_combinations = []
    for i, (model_type, encoder) in enumerate(combinations):
        if validate_combination(model_type, encoder, num_classes):
            valid_combinations.append((model_type, encoder))
            if i % 10 == 0:
                logger.info(f"已验证 {i + 1}/{total_combinations} 个组合")

    logger.info(f"有效组合: {len(valid_combinations)}/{total_combinations}")

    if resume_from:
        logger.info(f"从第 {resume_from} 个组合开始恢复训练")
        valid_combinations = valid_combinations[resume_from - 1:]

    results = {
        'timestamp': timestamp,
        'mode': mode,
        'dataset_type': Config.DATASET_TYPE,
        'total_combinations': total_combinations,
        'valid_combinations': len(valid_combinations),
        'combinations': [],
        'successful': [],
        'failed': [],
        'total_time': 0
    }

    start_time = time.time()

    for i, (model_type, encoder) in enumerate(valid_combinations, 1):
        combo_info = {
            'index': i,
            'model_type': model_type,
            'encoder': encoder,
            'status': 'pending'
        }
        results['combinations'].append(combo_info)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"训练组合 [{i}/{len(valid_combinations)}]: {model_type} + {encoder}")
        logger.info(f"{'=' * 60}")

        try:
            combo_start_time = time.time()

            with training_context(model_type, encoder):
                train_model()

            combo_end_time = time.time()
            training_duration = combo_end_time - combo_start_time

            success_info = {
                'index': i,
                'model_type': model_type,
                'encoder': encoder,
                'training_time': training_duration,
                'status': 'success'
            }
            results['successful'].append(success_info)
            combo_info['status'] = 'success'
            combo_info['training_time'] = training_duration

            logger.info(f"✓ {model_type} + {encoder} 训练完成 (用时: {training_duration:.2f}秒)")

            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在保存结果...")
            break

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ {model_type} + {encoder} 训练失败: {error_msg}")
            logger.error(f"详细错误: {traceback.format_exc()}")

            failure_info = {
                'index': i,
                'model_type': model_type,
                'encoder': encoder,
                'error': error_msg,
                'status': 'failed'
            }
            results['failed'].append(failure_info)
            combo_info['status'] = 'failed'
            combo_info['error'] = error_msg

            torch.cuda.empty_cache()
            continue

        if i % 5 == 0:
            temp_results_file = save_training_results(results, f"{timestamp}_temp")
            logger.info(f"临时结果已保存到: {temp_results_file}")

    end_time = time.time()
    results['total_time'] = end_time - start_time

    results_file = save_training_results(results, timestamp)
    logger.info(f"最终结果已保存到: {results_file}")

    generate_summary_report(results, logger)

    return results


def interactive_menu():
    print("\n" + "=" * 60)
    print("SMP模型批量训练脚本")
    print("=" * 60)

    print("\n选择训练模式:")
    print("1. 优先模式 - 训练主要的模型+编码器组合 (约100个)")
    print("2. 完整模式 - 训练所有可能的组合 (约600个)")
    print("3. 自定义模式 - 指定特定的模型和编码器")
    print("4. 恢复训练 - 从中断点继续")
    print("0. 退出")

    choice = input("\n请输入选择 (0-4): ").strip()

    if choice == '1':
        return 'priority', None
    elif choice == '2':
        return 'full', None
    elif choice == '3':
        return 'custom', None
    elif choice == '4':
        resume_from = input("请输入恢复的组合序号: ").strip()
        try:
            resume_from = int(resume_from)
            mode = input("请输入原始模式 (priority/full): ").strip()
            return mode, resume_from
        except ValueError:
            print("无效的序号")
            return None, None
    elif choice == '0':
        return None, None
    else:
        print("无效选择")
        return None, None


def custom_training():
    logger = setup_logging()

    print("\n自定义训练模式")
    print("可用模型类型:", get_all_model_types())
    print("可用编码器 (部分):", get_priority_encoders()[:10], "...")

    model_types = input("请输入模型类型 (用逗号分隔): ").strip().split(',')
    encoders = input("请输入编码器名称 (用逗号分隔): ").strip().split(',')

    model_types = [m.strip() for m in model_types if m.strip()]
    encoders = [e.strip() for e in encoders if e.strip()]

    if not model_types or not encoders:
        logger.error("模型类型和编码器不能为空")
        return

    dataset_config = Config.get_dataset_config()
    num_classes = dataset_config['num_classes']

    combinations = [(model, encoder) for model in model_types for encoder in encoders]
    valid_combinations = []

    for model_type, encoder in combinations:
        if validate_combination(model_type, encoder, num_classes):
            valid_combinations.append((model_type, encoder))
        else:
            logger.warning(f"组合 {model_type} + {encoder} 不可用")

    if not valid_combinations:
        logger.error("没有有效的组合可以训练")
        return

    logger.info(f"将训练 {len(valid_combinations)} 个组合")

    confirm = input("确认开始训练? (y/n): ").strip().lower()
    if confirm in ['y', 'yes', '是']:
        results = batch_train_all_combinations_custom(valid_combinations)
        return results


def batch_train_all_combinations_custom(combinations):
    logger = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'timestamp': timestamp,
        'mode': 'custom',
        'dataset_type': Config.DATASET_TYPE,
        'combinations': [],
        'successful': [],
        'failed': [],
        'total_time': 0
    }

    start_time = time.time()

    for i, (model_type, encoder) in enumerate(combinations, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"训练组合 [{i}/{len(combinations)}]: {model_type} + {encoder}")
        logger.info(f"{'=' * 60}")

        try:
            combo_start_time = time.time()

            with training_context(model_type, encoder):
                train_model()

            combo_end_time = time.time()
            training_duration = combo_end_time - combo_start_time

            success_info = {
                'model_type': model_type,
                'encoder': encoder,
                'training_time': training_duration
            }
            results['successful'].append(success_info)

            logger.info(f"✓ 训练完成 (用时: {training_duration:.2f}秒)")
            torch.cuda.empty_cache()

        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ 训练失败: {error_msg}")

            failure_info = {
                'model_type': model_type,
                'encoder': encoder,
                'error': error_msg
            }
            results['failed'].append(failure_info)
            torch.cuda.empty_cache()

    end_time = time.time()
    results['total_time'] = end_time - start_time

    results_file = save_training_results(results, timestamp)
    logger.info(f"结果已保存到: {results_file}")

    generate_summary_report(results, logger)
    return results


if __name__ == "__main__":
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("警告: 未检测到CUDA支持")

    mode, resume_from = interactive_menu()

    if mode is None:
        print("退出程序")
        sys.exit(0)

    if mode == 'custom':
        custom_training()
    else:
        print(f"\n开始{mode}模式批量训练...")
        if resume_from:
            print(f"从第 {resume_from} 个组合恢复")

        try:
            batch_train_all_combinations(mode, resume_from)
        except KeyboardInterrupt:
            print("\n用户中断训练")
        except Exception as e:
            print(f"\n训练过程出错: {e}")