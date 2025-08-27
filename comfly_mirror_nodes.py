"""
Comfly AI 镜像站节点
兼容Gemini API格式，但使用Comfly镜像服务地址
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import time
import random
from typing import Optional, Tuple, Dict, Any, List

try:
    from .tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    from .utils import (
        image_to_base64, base64_to_image,
        validate_api_key, format_error_message, resize_image_for_api
    )
    from .config import DEFAULT_CONFIG
except ImportError:
    # 修复导入路径，不使用前缀点，因为我们已经在comfyui_nano包中
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    except ImportError:
        print("⚠️ 无法导入tensor_utils模块，使用内置简化版本")
        # 简化版tensor转换函数
        def tensor_to_pil(tensor):
            """将tensor转换为PIL图像"""
            tensor = tensor.cpu().detach().numpy()
            # 将tensor的维度从[C, H, W]转换为[H, W, C]
            tensor = tensor.transpose(1, 2, 0)
            # 确保值的范围是0-1
            tensor = tensor * 255
            tensor = np.clip(tensor, 0, 255).astype(np.uint8)
            # 创建PIL图像
            return Image.fromarray(tensor)
        
        def pil_to_tensor(pil_image):
            """将PIL图像转换为tensor"""
            # 确保图像是RGB模式
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            # 转换为numpy数组
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            # 转换为[C, H, W]格式
            np_image = np_image.transpose(2, 0, 1)
            # 创建tensor
            return torch.from_numpy(np_image)
        
        def batch_tensor_to_pil_list(batch_tensor):
            """将批次tensor转换为PIL图像列表"""
            return [tensor_to_pil(batch_tensor[i]) for i in range(batch_tensor.shape[0])]
        
        def get_tensor_info(tensor):
            """获取tensor的信息"""
            shape = tensor.shape
            return f"形状:{shape}, 类型:{tensor.dtype}, 设备:{tensor.device}, 最小值:{tensor.min().item():.4f}, 最大值:{tensor.max().item():.4f}"
    
    # Fallback utility functions - 其他辅助函数的简化版本
    def image_to_base64(image, format='JPEG'):
        buffer = io.BytesIO()
        if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        image.save(buffer, format=format, quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def base64_to_image(base64_string):
        """Base64字符串转PIL图像"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    def validate_api_key(api_key):
        return api_key and len(api_key.strip()) > 10
    
    def format_error_message(error):
        return str(error)
    
    def resize_image_for_api(image, max_size=4000):
        """调整图像大小以满足API限制"""
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        
        # 计算缩放比例
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    DEFAULT_CONFIG = {"timeout": 120, "max_retries": 3}


def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟"""
    base_delay = 2 ** attempt
    
    if error_code == 429:
        rate_limit_delay = 60 + random.uniform(10, 30)
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:
        return base_delay + random.uniform(1, 5)
    else:
        return base_delay


class ComflyGeminiMirror:
    """Comfly镜像站 Gemini 图像编辑与生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "描述并编辑这些图像，或者生成新图像", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.5-flash-image-preview"}),
                "mode": (["edit", "generate"], {"default": "edit"}),  # 编辑或生成模式
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "response_text")
    FUNCTION = "process"
    CATEGORY = "Nano"

    def process(self, api_key, prompt, model, mode, seed=0, temperature=1.0, top_p=0.95, max_output_tokens=8192, images=None):
        """处理请求，根据模式进行编辑或生成"""
        
        # 检查API密钥
        if not validate_api_key(api_key):
            raise ValueError("请提供有效的API密钥")
        
        # 根据模式决定处理方法
        if mode == "edit":
            if images is None:
                raise ValueError("编辑模式下需要提供输入图像")
            return self._process_edit(api_key, images, prompt, model, seed, temperature, top_p, max_output_tokens)
        else:  # generate
            return self._process_generate(api_key, prompt, model, seed, temperature, top_p, max_output_tokens)
    
    def _process_edit(self, api_key, images, prompt, model, seed, temperature, top_p, max_output_tokens):
        """处理图像编辑请求"""
        
        # 将批次图像转换为PIL图像列表
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
        print(f"📥 收到 {len(pil_images)} 张图像进行编辑")
        
        # 构建包含多张图像的请求
        parts = [{"text": prompt.strip()}]
        
        # 添加所有图像
        for i, pil_image in enumerate(pil_images):
            image_base64 = image_to_base64(pil_image, format='JPEG')
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
            print(f"📎 添加第 {i+1} 张图像到请求中")
        
        # 构建API URL - 使用Comfly镜像站地址
        url = f"https://ai.comfly.chat/v1beta/models/{model}:generateContent"
        
        # 构建请求数据
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, pil_images[0], model)
    
    def _process_generate(self, api_key, prompt, model, seed, temperature, top_p, max_output_tokens):
        """处理图像生成请求"""
        
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        print(f"ℹ️ 使用种子 {seed}, 但注意 Gemini API 当前不支持种子参数")
        
        # 构建API URL - 使用Comfly镜像站地址
        url = f"https://ai.comfly.chat/v1beta/models/{model}:generateContent"
        
        request_data = {
            "contents": [{
                "parts": [
                    {"text": prompt.strip()}
                ]
            }],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key.strip()
        }
        
        # 创建一个默认图像作为fallback
        default_image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        return self._send_request_and_process(url, headers, request_data, default_image, model)
    
    def _send_request_and_process(self, url, headers, request_data, fallback_image, model):
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在处理请求... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                print(f"🌐 使用Comfly镜像站: {url}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应和图片
                    response_text = ""
                    processed_images = []
                    
                    if "candidates" in result and result["candidates"]:
                        for candidate in result["candidates"]:
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    # 提取文本
                                    if "text" in part:
                                        response_text += part["text"]
                                    
                                    # 提取图片
                                    if "inline_data" in part or "inlineData" in part:
                                        inline_data = part.get("inline_data") or part.get("inlineData")
                                        if inline_data and "data" in inline_data:
                                            try:
                                                image_data = inline_data["data"]
                                                image_bytes = base64.b64decode(image_data)
                                                processed_image = Image.open(io.BytesIO(image_bytes))
                                                processed_images.append(processed_image)
                                                print("✅ 成功提取图片")
                                            except Exception as e:
                                                print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有处理后的图片，返回原图片
                    if not processed_images:
                        print("⚠️ 未检测到生成/编辑后的图片，返回原图片")
                        processed_images.append(fallback_image)
                        if not response_text:
                            response_text = "请求已发送，但未收到生成/编辑后的图片"
                    
                    # 转换为tensor
                    if len(processed_images) == 1:
                        image_tensor = pil_to_tensor(processed_images[0])
                    else:
                        # 多张图片时，创建批次tensor
                        tensors = [pil_to_tensor(img) for img in processed_images]
                        image_tensor = torch.stack(tensors, dim=0)
                    
                    print(f"✅ 处理完成，输出张量形状: {image_tensor.shape}")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    delay = smart_retry_delay(attempt, response.status_code)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                error_msg = format_error_message(e)
                print(f"❌ 请求失败: {error_msg}")
                if attempt == max_retries - 1:
                    raise ValueError(f"API请求失败: {error_msg}")
                else:
                    delay = smart_retry_delay(attempt)
                    print(f"🔄 等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = format_error_message(e)
                print(f"❌ 处理失败: {error_msg}")
                raise ValueError(f"处理失败: {error_msg}")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ComflyGeminiMirror": ComflyGeminiMirror,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComflyGeminiMirror": "Comfly镜像站",
} 