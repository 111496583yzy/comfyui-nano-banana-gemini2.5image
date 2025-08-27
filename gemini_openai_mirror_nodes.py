"""
Gemini OpenAI 格式镜像站节点
支持使用 OpenAI Chat Completions API 格式的镜像站进行图像生成和编辑
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
from typing import Optional, Tuple, Dict, Any
import re

try:
    from .utils import (
        tensor_to_pil, pil_to_tensor, image_to_base64, base64_to_image,
        validate_api_key, format_error_message, resize_image_for_api
    )
    from .config import DEFAULT_CONFIG
except ImportError:
    # Fallback utility functions
    def tensor_to_pil(tensor):
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.cpu().numpy())
    
    def pil_to_tensor(image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).unsqueeze(0)
        return tensor
    
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


def validate_api_url(url):
    """验证API URL格式"""
    if not url or not url.strip():
        return False
    
    url = url.strip()
    # 基本URL格式检查
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def build_openai_api_url(base_url):
    """构建 OpenAI 格式的 API URL"""
    base_url = base_url.strip().rstrip('/')
    
    # 如果用户提供的是完整URL，直接使用
    if '/chat/completions' in base_url:
        return base_url
    
    # 如果是基础URL，构建完整路径
    if base_url.endswith('/v1'):
        return f"{base_url}/chat/completions"
    
    # 默认添加v1路径
    return f"{base_url}/v1/chat/completions"


def smart_retry_delay(attempt, error_code=None):
    """智能重试延迟 - 根据错误类型调整等待时间"""
    base_delay = 2 ** attempt  # 指数退避
    
    if error_code == 429:  # 限流错误
        rate_limit_delay = 60 + random.uniform(10, 30)  # 60-90秒随机等待
        return max(base_delay, rate_limit_delay)
    elif error_code in [500, 502, 503, 504]:  # 服务器错误
        return base_delay + random.uniform(1, 5)  # 添加随机抖动
    else:
        return base_delay


def extract_image_from_openai_response(response_data):
    """从 OpenAI 格式响应中提取图片"""
    try:
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                
                # 如果内容是字符串，尝试解析为JSON
                if isinstance(content, str):
                    try:
                        content_json = json.loads(content)
                        if isinstance(content_json, dict) and "image" in content_json:
                            image_data = content_json["image"]
                            if image_data.startswith("data:image"):
                                # 处理 data URL 格式
                                header, data = image_data.split(",", 1)
                                image_bytes = base64.b64decode(data)
                                return Image.open(io.BytesIO(image_bytes))
                            else:
                                # 直接的 base64 数据
                                image_bytes = base64.b64decode(image_data)
                                return Image.open(io.BytesIO(image_bytes))
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                # 检查是否有图片URL或base64数据
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "type" in item and item["type"] == "image_url":
                                if "image_url" in item and "url" in item["image_url"]:
                                    url = item["image_url"]["url"]
                                    if url.startswith("data:image"):
                                        header, data = url.split(",", 1)
                                        image_bytes = base64.b64decode(data)
                                        return Image.open(io.BytesIO(image_bytes))
                            elif "image" in item:
                                image_data = item["image"]
                                image_bytes = base64.b64decode(image_data)
                                return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"⚠️ 解析图片数据失败: {e}")
    
    return None


class GeminiOpenAIMirrorImageGeneration:
    """Gemini OpenAI 格式镜像站图片生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://ai.t8star.cn", 
                    "multiline": False,
                    "placeholder": "输入OpenAI格式API地址，如: https://ai.t8star.cn"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "A beautiful mountain landscape at sunset", "multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-image", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "Gemini/OpenAI Mirror"
    
    def generate_image(self, api_url: str, api_key: str, prompt: str, model: str, 
                      temperature: float, max_tokens: int) -> Tuple[torch.Tensor, str]:
        """使用 OpenAI 格式镜像站生成图片"""
        
        # 验证API URL
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请输入有效的URL地址")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 构建完整的API URL
        full_url = build_openai_api_url(api_url)
        print(f"🌐 使用OpenAI格式API地址: {full_url}")
        
        # 构建 OpenAI 格式请求数据
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"生成图片: {prompt.strip()}"
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        
        # 智能重试机制
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 提示词: {prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                print(f"🤖 模型: {model}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    # 解析响应
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应
                    response_text = ""
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"]
                    
                    # 尝试从响应中提取图片
                    generated_image = extract_image_from_openai_response(result)
                    
                    # 如果没有生成图片，创建占位符
                    if generated_image is None:
                        print("⚠️ 未检测到生成的图片，创建占位符")
                        generated_image = Image.new('RGB', (512, 512), color='lightgray')
                        if not response_text:
                            response_text = "图片生成请求已发送，但未收到图片数据。可能需要调整模型或提示词格式。"
                        else:
                            response_text += "\n\n注意: 未检测到图片数据，显示占位符。"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(generated_image)
                    
                    print("✅ 图片生成完成")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    # 智能等待
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
                raise ValueError(f"图片生成失败: {error_msg}")


class GeminiOpenAIMirrorImageEdit:
    """Gemini OpenAI 格式镜像站图片编辑节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://ai.t8star.cn", 
                    "multiline": False,
                    "placeholder": "输入OpenAI格式API地址，如: https://ai.t8star.cn"
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "修改这个图片，让模特穿上时尚的服装", "multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-image", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "edit_image"
    CATEGORY = "Gemini/OpenAI Mirror"
    
    def edit_image(self, api_url: str, api_key: str, image: torch.Tensor, prompt: str, model: str,
                   temperature: float, max_tokens: int) -> Tuple[torch.Tensor, str]:
        """使用 OpenAI 格式镜像站编辑图片"""
        
        # 验证API URL
        if not validate_api_url(api_url):
            raise ValueError("API URL格式无效，请输入有效的URL地址")
        
        # 验证API密钥
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("提示词不能为空")
        
        # 转换输入图片
        pil_image = tensor_to_pil(image)
        
        # 转换为base64
        image_base64 = image_to_base64(pil_image, format='JPEG')
        
        # 构建完整的API URL
        full_url = build_openai_api_url(api_url)
        print(f"🌐 使用OpenAI格式API地址: {full_url}")
        
        # 构建 OpenAI 格式请求数据 - 包含文本和图片
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.strip()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        
        # 智能重试机制
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🖼️ 正在编辑图片... (尝试 {attempt + 1}/{max_retries})")
                print(f"📝 编辑指令: {prompt[:100]}...")
                print(f"🔗 镜像站: {api_url}")
                print(f"🤖 模型: {model}")
                
                # 发送请求
                response = requests.post(full_url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    # 解析响应
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取文本响应
                    response_text = ""
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"]
                    
                    # 尝试从响应中提取编辑后的图片
                    edited_image = extract_image_from_openai_response(result)
                    
                    # 如果没有编辑后的图片，返回原图片
                    if edited_image is None:
                        print("⚠️ 未检测到编辑后的图片，返回原图片")
                        edited_image = pil_image
                        if not response_text:
                            response_text = "图片编辑请求已发送，但未收到编辑后的图片数据。可能需要调整模型或提示词格式。"
                        else:
                            response_text += "\n\n注意: 未检测到编辑后的图片数据，返回原图片。"
                    
                    # 转换为tensor
                    image_tensor = pil_to_tensor(edited_image)
                    
                    print("✅ 图片编辑完成")
                    return (image_tensor, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"❌ 错误文本: {response.text}")
                    
                    # 如果是最后一次尝试，抛出异常
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    
                    # 智能等待
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
                raise ValueError(f"图片编辑失败: {error_msg}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiOpenAIMirrorImageGeneration": GeminiOpenAIMirrorImageGeneration,
    "GeminiOpenAIMirrorImageEdit": GeminiOpenAIMirrorImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiOpenAIMirrorImageGeneration": "Gemini OpenAI格式镜像站图片生成",
    "GeminiOpenAIMirrorImageEdit": "Gemini OpenAI格式镜像站图片编辑",
}