"""
图像处理节点
包含高反差保留等图像增强功能
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple

try:
    from .tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info


class HighPassFilterNode:
    """
    PS高反差保留节点
    实现高反差保留滤镜效果，用于增强图像细节和对比度
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 输入的图像批次
                "radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "高斯模糊半径，控制保留的细节范围"
                }),
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "高反差保留的强度，0=无效果，1=标准效果，大于1=增强效果"
                }),
                "blend_mode": (["normal", "overlay", "soft_light"], {
                    "default": "normal",
                    "tooltip": "混合模式：normal=正常叠加，overlay=叠加模式，soft_light=柔光模式"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("high_pass_images",)
    FUNCTION = "apply_high_pass"
    CATEGORY = "Nano/图像处理"
    
    def apply_high_pass(self, images, radius, amount, blend_mode):
        """
        应用高反差保留滤镜
        
        Args:
            images: 输入图像批次 [B, H, W, C] 或 [B, C, H, W]
            radius: 高斯模糊半径
            amount: 高反差强度
            blend_mode: 混合模式
            
        Returns:
            处理后的图像批次
        """
        print(f"🔄 应用高反差保留滤镜: radius={radius}, amount={amount}, blend_mode={blend_mode}")
        
        # 确保输入是tensor格式
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # 处理批次维度
        batch_size = images.shape[0]
        print(f"📊 输入图像批次大小: {batch_size}, 图像形状: {images.shape}")
        
        # 转换tensor为PIL图像列表
        pil_images = batch_tensor_to_pil_list(images)
        
        processed_images = []
        
        for i, pil_image in enumerate(pil_images):
            try:
                # 应用高反差保留
                result = self._high_pass_filter(pil_image, radius, amount, blend_mode)
                processed_images.append(result)
                
            except Exception as e:
                print(f"⚠️ 处理第 {i+1} 张图像时出错: {e}")
                # 出错时返回原图
                processed_images.append(pil_image)
        
        # 转换回tensor并保持正确的批次格式
        result_tensors = []
        for img in processed_images:
            # pil_to_tensor 返回 (1, H, W, C)，去掉批次维度
            tensor = pil_to_tensor(img)
            result_tensors.append(tensor.squeeze(0))  # 去掉批次维度
        
        # 堆叠为批次 (batch, H, W, C)
        result_batch = torch.stack(result_tensors, dim=0)
        
        print(f"✅ 高反差保留处理完成，输出批次大小: {result_batch.shape}")
        return (result_batch,)
    
    def _high_pass_filter(self, image, radius, amount, blend_mode):
        """
        对单张图像应用高反差保留滤镜
        
        算法原理：
        1. 原图像 -> 高斯模糊 -> 模糊图
        2. 原图 - 模糊图 = 细节图
        3. 原图 + 细节图 * amount = 最终结果
        """
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用高斯模糊
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # 转换为numpy数组进行计算
        original = np.array(image, dtype=np.float32)
        blurred_array = np.array(blurred, dtype=np.float32)
        
        # 计算高反差保留：原图 - 模糊图 = 细节图
        # 将细节叠加到中灰度值上
        details = original - blurred_array + 128.0
        
        # 应用强度参数
        # 通过调整原图和细节图的混合比例来实现
        result = blurred_array + (details - blurred_array) * amount
        
        # 限制像素值在有效范围内
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 根据混合模式调整结果
        if blend_mode == "normal":
            # 正常模式：直接使用高反差保留结果
            output = Image.fromarray(result)
        elif blend_mode == "overlay":
            # 叠加模式：模拟PS的overlay混合
            output = self._blend_overlay(image, blurred, amount)
        elif blend_mode == "soft_light":
            # 柔光模式：更柔和的混合效果
            output = self._blend_soft_light(image, blurred, amount)
        else:
            output = Image.fromarray(result)
        
        return output
    
    def _blend_overlay(self, original, blurred, amount):
        """叠加混合模式"""
        orig_array = np.array(original, dtype=np.float32)
        blur_array = np.array(blurred, dtype=np.float32)
        
        # 计算高反差保留
        high_pass = orig_array - blur_array + 128.0
        high_pass = np.clip(high_pass, 0, 255)
        
        # 叠加混合公式
        mask = (orig_array < 128).astype(np.float32)
        result = mask * (2 * orig_array * high_pass / 255.0) + \
                (1 - mask) * (255 - 2 * (255 - orig_array) * (255 - high_pass) / 255.0)
        
        # 应用强度
        result = orig_array * (1 - amount) + result * amount
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    
    def _blend_soft_light(self, original, blurred, amount):
        """柔光混合模式"""
        orig_array = np.array(original, dtype=np.float32)
        blur_array = np.array(blurred, dtype=np.float32)
        
        # 计算高反差保留
        high_pass = orig_array - blur_array + 128.0
        high_pass = np.clip(high_pass, 0, 255)
        
        # 柔光混合公式
        mask = (orig_array < 128).astype(np.float32)
        result = mask * (2 * orig_array * high_pass / 255.0) + \
                (1 - mask) * (orig_array + (2 * high_pass - 255) * (255 - orig_array) / 255.0)
        
        # 应用强度
        result = orig_array * (1 - amount) + result * amount
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "HighPassFilterNode": HighPassFilterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighPassFilterNode": "✨ 高反差保留",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

