import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from PIL import Image


class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        初始化UNet模型
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        """
        super().__init__()
        
        # 编码器部分
        self.conv1 = DoubleConv(in_channels, 32)  # 减少特征通道数以降低内存占用
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        
        # 解码器部分
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(64, 32)
        
        # 输出层
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 模型输出
        """
        # 编码器路径
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        
        # 解码器路径（含跳跃连接）
        up_1 = self.up1(c5)
        merge1 = torch.cat([up_1, c4], dim=1)
        up_conv1 = self.up_conv1(merge1)
        up_2 = self.up2(up_conv1)
        merge2 = torch.cat([up_2, c3], dim=1)
        up_conv2 = self.up_conv2(merge2)
        up_3 = self.up3(up_conv2)
        merge3 = torch.cat([up_3, c2], dim=1)
        up_conv3 = self.up_conv3(merge3)
        up_4 = self.up4(up_conv3)
        merge4 = torch.cat([up_4, c1], dim=1)
        up_conv4 = self.up_conv4(merge4)
        
        # 输出处理
        out = self.out(up_conv4)
        return torch.sigmoid(out).squeeze(1)
    

class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 模块
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 
                              padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 
                              padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 
                              padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], 
                          mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn_out(self.conv_out(x)))
        
        return x
    
class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 模型实现
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super(DeepLabV3Plus, self).__init__()
        
        # 编码器初始层
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 编码器主干网络
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # ASPP模块
        self.aspp = ASPP(512, 256)
        
        # 低层特征处理
        self.low_level_conv = nn.Conv2d(64, 48, 1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 
                                  padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size()[2:]
        
        # 编码器路径
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        low_level_feat = x
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP处理
        x = self.aspp(x)
        
        # 上采样到低层特征的尺寸
        x = F.interpolate(x, size=low_level_feat.size()[2:], 
                         mode='bilinear', align_corners=True)
        
        # 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        print(f"Low-level feature shape: {low_level_feat.shape}") 
        # 特征融合
        x = torch.cat([x, low_level_feat], dim=1)
        
        # 解码器处理
        x = self.decoder(x)
        
        # 上采样到原始输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # sigmoid激活
        out = self.sigmoid(x)
        
        return out.squeeze(1) if out.shape[1] == 1 else out


def preprocess_image(image, target_size=(400, 400)):
    """
    图像预处理：调整大小、归一化、转换为张量
    Args:
        image: 输入图像，PIL.Image 或 numpy.ndarray 格式
        target_size: 目标输入大小 (height, width)
    Returns:
        处理后的图像张量
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 转换为 RGB 格式
    if len(image.shape) == 2:  # 灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA 图
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 预处理操作
    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 的均值
            std=[0.229, 0.224, 0.225],   # ImageNet 的标准差
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    # 应用预处理
    transformed = transform(image=image)
    return transformed['image']


def predict_mask_unet(model, image, device):
    """
    使用 UNet 模型进行预测
    Args:
        model: UNet 模型
        image: 输入图像，PIL.Image 或 numpy.ndarray 格式
        device: 运行设备（'cpu' 或 'cuda'）
    Returns:
        分割掩码，numpy.ndarray 格式，值为 0 或 255
    """
    model.eval()
    original_size = image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]
    
    # 图像预处理
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.squeeze().cpu().numpy()  # 转为 numpy 格式
    
    # 恢复到原始尺寸并二值化
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # 二值化
    return pred_mask

def predict_mask_deeplabv3(model, image, device):
    """
    使用 DeepLabV3+ 模型进行预测
    Args:
        model: DeepLabV3+ 模型
        image: 输入图像，PIL.Image 或 numpy.ndarray 格式
        device: 运行设备（'cpu' 或 'cuda'）
    Returns:
        分割掩码，numpy.ndarray 格式，值为 0 或 255
    """
    model.eval()
    original_size = image.size if isinstance(image, Image.Image) else image.shape[:2][::-1]
    
    # 图像预处理
    input_tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.squeeze().cpu().numpy()
    
    # 恢复到原始尺寸并二值化
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # 二值化
    return pred_mask

def predict_mask(model, image, device):
    """
    统一的预测接口，根据模型类型调用不同的预测函数
    Args:
        model: UNet 或 DeepLabV3+ 模型
        image: 输入图像，PIL.Image 或 numpy.ndarray 格式
        device: 运行设备（'cpu' 或 'cuda'）
    Returns:
        分割掩码，numpy.ndarray 格式，值为 0 或 255
    """
    try:
        model_name = model.__class__.__name__
        if model_name == 'UNet':
            return predict_mask_unet(model, image, device)
        elif model_name == 'DeepLabV3Plus':
            return predict_mask_deeplabv3(model, image, device)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    except Exception as e:
        print(f"Error in predict_mask: {str(e)}")
        raise