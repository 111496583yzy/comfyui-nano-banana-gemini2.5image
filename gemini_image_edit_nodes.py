"""
Gemini 图像编辑节点
支持单图和多图输入，自动处理批次数据
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
    from .tensor_utils import tensor_to_pil, pil_to_tensor, batch_tensor_to_pil_list, get_tensor_info
    # Fallback utility functions - 如果无法导入，使用内置版本
    pass
    
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
    
    def validate_api_key(api_key):
        return api_key and len(api_key.strip()) > 10
    
    def format_error_message(error):
        return str(error)
    
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


class GeminiImageEdit:
    """Gemini 图像编辑节点 - 使用优化的批量处理方式避免白边问题"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "images": ("IMAGE",),  # 支持批次图像
                "prompt": ("STRING", {"default": "Describe these images and edit them", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.0-flash-preview-image-generation"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "process_images"
    CATEGORY = "Nano"

    def process_images(self, api_key, images, prompt, model, seed=0, temperature=1.0, top_p=0.95, max_output_tokens=8192):
        """处理图像并返回编辑后的图像和响应文本"""
        
        # 检查API密钥
        if not api_key:
            raise ValueError("请提供有效的Gemini API密钥")
        
        # 将批次图像转换为PIL图像列表
        pil_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
        print(f"📥 收到 {len(pil_images)} 张图像进行处理")
        
        # 使用优化的批量处理模式处理图像
        print(f"🔄 使用优化的批量处理模式处理图像")
        print(f"ℹ️ Received seed {seed}, but the Gemini API does not currently support a seed parameter for image editing.")
        edited_tensor, response_text = self._process_combined_images(api_key, pil_images, prompt, model, temperature, top_p, max_output_tokens)
        
        return (edited_tensor, response_text)
    
    def _process_combined_images(self, api_key: str, pil_images: List[Image.Image], prompt: str, model: str,
                                temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        """处理多张图像（合并发送）"""
        
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
        
        # 构建API URL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # 构建请求数据 - 更新为匹配官方示例的格式
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
    
    def _send_request_and_process(self, url: str, headers: dict, request_data: dict, 
                                 fallback_image: Image.Image, model: str) -> Tuple[torch.Tensor, str]:
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在处理图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                # 提取文本
                                if "text" in part:
                                    response_text += part["text"]
                                
                                # 提取编辑后的图片
                                if "inline_data" in part or "inlineData" in part:
                                    inline_data = part.get("inline_data") or part.get("inlineData")
                                    if inline_data and "data" in inline_data:
                                        try:
                                            image_data = inline_data["data"]
                                            image_bytes = base64.b64decode(image_data)
                                            edited_image = Image.open(io.BytesIO(image_bytes))
                                            print("✅ 成功提取编辑后的图片")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = fallback_image
                        if not response_text:
                            response_text = "图片处理请求已发送，但未收到编辑后的图片"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 图片处理完成")
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
                raise ValueError(f"图片处理失败: {error_msg}")


class GeminiImageGenerate:
    """Gemini 图像生成节点 - 根据文本提示生成图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-preview-image-generation"], {"default": "gemini-2.5-flash-image-preview"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response_text")
    FUNCTION = "generate_images"
    CATEGORY = "Nano"

    def generate_images(self, api_key: str, prompt: str, model: str, seed: int,
                        temperature: float, top_p: float, max_output_tokens: int) -> Tuple[torch.Tensor, str]:
        
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        if not prompt.strip():
            raise ValueError("提示词不能为空")

        print(f"ℹ️ Received seed {seed}, but the Gemini API does not currently support a seed parameter for image generation.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
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
        
        return self._send_request_and_generate_images(url, headers, request_data, model)

    def _send_request_and_generate_images(self, url: str, headers: dict, request_data: dict, model: str) -> Tuple[torch.Tensor, str]:
        """发送生成请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    generated_images = []
                    response_texts = []
                    
                    if "candidates" in result and result["candidates"]:
                        for i, candidate in enumerate(result["candidates"]):
                            candidate_text = ""
                            candidate_image = None
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        candidate_text += part["text"]
                                    
                                    if "inline_data" in part or "inlineData" in part:
                                        inline_data = part.get("inline_data") or part.get("inlineData")
                                        if inline_data and "data" in inline_data:
                                            try:
                                                image_data = inline_data["data"]
                                                image_bytes = base64.b64decode(image_data)
                                                candidate_image = Image.open(io.BytesIO(image_bytes))
                                            except Exception as e:
                                                print(f"⚠️ 解码候选图片 {i+1} 失败: {e}")
                            
                            if candidate_image:
                                generated_images.append(candidate_image)
                                response_texts.append(f"图像 {i+1}:\n{candidate_text}")
                                print(f"✅ 成功提取生成的图片 {i+1}")

                    if not generated_images:
                        raise ValueError("API响应中未找到有效的生成图片")
                    
                    if len(generated_images) == 1:
                        final_tensor = pil_to_tensor(generated_images[0])
                    else:
                        tensors = [pil_to_tensor(img) for img in generated_images]
                        final_tensor = torch.stack(tensors, dim=0)
                    
                    combined_response = "\n\n".join(response_texts)
                    print(f"✅ 图像生成完成，输出张量形状: {final_tensor.shape}")
                    return (final_tensor, combined_response)
                
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
                print(f"❌ 生成失败: {error_msg}")
                raise ValueError(f"图像生成失败: {error_msg}")

# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiImageEdit": GeminiImageEdit,
    "GeminiImageGenerate": GeminiImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEdit": "Gemini 图像编辑",
    "GeminiImageGenerate": "Gemini 图像生成",
}