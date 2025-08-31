import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    def __init__(self, model_type="DeepLabV3", num_classes=4, encoder_name="resnet101", encoder_weights=None):
        super(SegmentationModel, self).__init__()

        # 支持的模型架构映射
        model_dict = {
            "DeepLabV3": smp.DeepLabV3,
            "DeepLabV3Plus": smp.DeepLabV3Plus,
            "UNet": smp.Unet,
            "UNetPlusPlus": smp.UnetPlusPlus,
            "FPN": smp.FPN,
            "PSPNet": smp.PSPNet,
            "Linknet": smp.Linknet,
            "MAnet": smp.MAnet,
            "PAN": smp.PAN,
            "Segformer": smp.Segformer
        }

        if model_type not in model_dict:
            available_models = list(model_dict.keys())
            raise ValueError(f"不支持的模型类型: {model_type}. 可用模型: {available_models}")

        self.model_type = model_type
        self.model = model_dict[model_type](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def get_model_info(self):
        """返回模型信息"""
        return {
            "model_type": self.model_type,
            "encoder": self.model.encoder.name if hasattr(self.model.encoder, 'name') else "unknown",
            "num_classes": self.model.segmentation_head[-1].out_channels
        }
