import os

class Config:
    # 数据集配置
    DATASET_TYPE = "KSDD2"  # 可选: "severstal", "KSDD2"

    # 模型配置
    MODEL_TYPE = "UNetPlusPlus"  # 可选: "DeepLabV3", "UNet", "FPN", "PSPNet", "Linknet"
    ENCODER_NAME = "densenet121"  # 可选
    ENCODER_WEIGHTS = None

    USER_RISK_LEVEL_A = 0.2


    # Severstal 数据集配置
    SEVERSTAL_CSV_FILE = "./data/severstal-steel-defect-detection/train.csv"
    SEVERSTAL_IMAGE_DIR = "./data/train/train_images"
    SEVERSTAL_IMAGE_SIZE = (256, 1600)
    SEVERSTAL_NUM_CLASSES = 4

    # KSDD2 数据集配置
    KSDD2_DATA_DIR = "./data/KolektorSDD2/train"
    KSDD2_IMAGE_SIZE = (256, 672)
    KSDD2_NUM_CLASSES = 1

    # 训练参数
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150
    NUM_WORKERS = 4

    # 数据分割
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.2

    # 检测参数
    DETECTION_THRESHOLD = 0.5

    # 输出路径
    OUTPUT_DIR = "./output"
    VISUALIZATION_DIR = "./visualization"

    # 模型保存路径
    @classmethod
    def get_model_path(cls):
        return f"fault_detection_{cls.MODEL_TYPE}_{cls.DATASET_TYPE}_{cls.ENCODER_NAME}.pth"

    @classmethod
    def get_output_path(cls):
        return os.path.join(cls.OUTPUT_DIR, f"output_{cls.DATASET_TYPE}_{cls.MODEL_TYPE}_{cls.ENCODER_NAME}.pt")

    @classmethod
    def get_visualization_dir(cls):
        return f"{cls.VISUALIZATION_DIR}_{cls.DATASET_TYPE}"

    @classmethod
    def get_dataset_config(cls):
        if cls.DATASET_TYPE == "severstal":
            return {
                'csv_file': cls.SEVERSTAL_CSV_FILE,
                'image_dir': cls.SEVERSTAL_IMAGE_DIR,
                'image_size': cls.SEVERSTAL_IMAGE_SIZE,
                'num_classes': cls.SEVERSTAL_NUM_CLASSES,
                'data_dir': None
            }
        elif cls.DATASET_TYPE == "KSDD2":
            return {
                'data_dir': cls.KSDD2_DATA_DIR,
                'image_size': cls.KSDD2_IMAGE_SIZE,
                'num_classes': cls.KSDD2_NUM_CLASSES,
                'csv_file': None,
                'image_dir': None
            }
        else:
            raise ValueError(f"不支持的数据集类型: {cls.DATASET_TYPE}")