"""
OpenRouter 镜像站节点
支持多种AI模型的统一API接口
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


class OpenRouterMirror:
    """OpenRouter 镜像站多模型AI节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "分析这张图像并描述其内容", "multiline": True}),
                "model": ([
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini", 
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3-haiku",
                    "google/gemini-pro-1.5",
                    "google/gemini-flash-1.5",
                    "meta-llama/llama-3.2-90b-vision-instruct",
                    "qwen/qwen-2-vl-72b-instruct",
                    "microsoft/phi-3.5-vision-instruct"
                ], {"default": "openai/gpt-4o-mini"}),
                "max_tokens": ("INT", {"default": 6664, "min": 1, "max": 6664}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "site_url": ("STRING", {"default": "", "multiline": False}),
                "app_name": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "mirror_url": ("STRING", {"default": "https://openrouter.ai", "multiline": False, "placeholder": "镜像站地址，默认为OpenRouter官方"}),
            }
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "model_info")
    FUNCTION = "process"
    CATEGORY = "Nano"

    def process(self, api_key, prompt, model, max_tokens=1024, temperature=0.7, top_p=1.0, 
                images=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None, 
                site_url="", app_name="ComfyUI", mirror_url="https://openrouter.ai"):
        """处理OpenRouter API请求"""
        
        # 检查API密钥
        if not validate_api_key(api_key):
            raise ValueError("请提供有效的OpenRouter API密钥")
        
        # 构建消息内容
        message_content = []
        
        # 添加文本提示
        message_content.append({
            "type": "text",
            "text": prompt.strip()
        })
        
        # 收集所有图像
        all_images = []
        
        # 处理批次图像（向后兼容）
        if images is not None:
            batch_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
            all_images.extend(batch_images)
            print(f"📥 从批次图像收到 {len(batch_images)} 张图像")
        
        # 处理6个独立的图像输入
        individual_images = [image_1, image_2, image_3, image_4, image_5, image_6]
        image_names = ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6"]
        
        for i, img in enumerate(individual_images):
            if img is not None:
                pil_image = tensor_to_pil(img)
                all_images.append(pil_image)
                print(f"📥 收到 {image_names[i]}: {pil_image.size}")
        
        # 如果有图像，添加到消息中
        if all_images:
            print(f"📥 收到 {len(all_images)} 张图像进行分析")
            
            for i, pil_image in enumerate(all_images):
                # 调整图像大小以满足API限制
                resized_image = resize_image_for_api(pil_image, max_size=2048)
                image_base64 = image_to_base64(resized_image, format='JPEG')
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })
                print(f"📎 添加第 {i+1} 张图像到请求中")
        
        # 构建API URL - 使用可配置的镜像站地址
        # 确保URL格式正确，移除末尾的斜杠
        base_url = mirror_url.rstrip('/')
        url = f"{base_url}/api/v1/chat/completions"
        
        # 构建请求数据
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url if site_url else "https://github.com/comfyanonymous/ComfyUI",
            "X-Title": app_name
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, model)
    
    def _send_request_and_process(self, url, headers, request_data, model):
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🤖 正在处理请求... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                print(f"🌐 使用OpenRouter API: {url}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取响应文本
                    response_text = ""
                    model_info = ""
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"]
                    
                    # 提取模型使用信息
                    if "usage" in result:
                        usage = result["usage"]
                        model_info = f"模型: {model}\n"
                        model_info += f"输入tokens: {usage.get('prompt_tokens', 'N/A')}\n"
                        model_info += f"输出tokens: {usage.get('completion_tokens', 'N/A')}\n"
                        model_info += f"总tokens: {usage.get('total_tokens', 'N/A')}"
                    
                    # 如果有模型ID信息
                    if "model" in result:
                        model_info += f"\n实际使用模型: {result['model']}"
                    
                    if not response_text:
                        response_text = "请求已发送，但未收到有效响应"
                    
                    print(f"✅ 处理完成，响应长度: {len(response_text)} 字符")
                    return (response_text, model_info)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                        
                        # 提取具体错误信息
                        if "error" in error_detail:
                            error_msg = error_detail["error"].get("message", str(error_detail["error"]))
                        else:
                            error_msg = str(error_detail)
                            
                    except:
                        error_msg = response.text
                        print(f"❌ 错误文本: {error_msg}")
                    
                    if attempt == max_retries - 1:
                        raise ValueError(f"OpenRouter API错误 ({response.status_code}): {error_msg}")
                    
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


class OpenRouterTextGeneration:
    """OpenRouter 纯文本生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "你好，请介绍一下你自己", "multiline": True}),
                "model": ([
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "openai/gpt-3.5-turbo",
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3-haiku",
                    "google/gemini-pro-1.5",
                    "google/gemini-flash-1.5",
                    "meta-llama/llama-3.1-405b-instruct",
                    "meta-llama/llama-3.1-70b-instruct",
                    "meta-llama/llama-3.1-8b-instruct",
                    "mistralai/mistral-7b-instruct",
                    "qwen/qwen-2-72b-instruct"
                ], {"default": "openai/gpt-4o-mini"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 6664}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "site_url": ("STRING", {"default": "", "multiline": False}),
                "app_name": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "mirror_url": ("STRING", {"default": "https://openrouter.ai", "multiline": False, "placeholder": "镜像站地址，默认为OpenRouter官方"}),
            }
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "model_info")
    FUNCTION = "generate"
    CATEGORY = "Nano"

    def generate(self, api_key, prompt, model, max_tokens=1024, temperature=0.7, top_p=1.0,
                seed=0, system_prompt="", site_url="", app_name="ComfyUI", mirror_url="https://openrouter.ai"):
        """生成文本响应"""
        
        # 检查API密钥
        if not validate_api_key(api_key):
            raise ValueError("请提供有效的OpenRouter API密钥")
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示（如果有）
        if system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt.strip()
            })
        
        # 添加用户提示
        messages.append({
            "role": "user", 
            "content": prompt.strip()
        })
        
        # 构建API URL - 使用可配置的镜像站地址
        # 确保URL格式正确，移除末尾的斜杠
        base_url = mirror_url.rstrip('/')
        url = f"{base_url}/api/v1/chat/completions"
        
        # 构建请求数据
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url if site_url else "https://github.com/comfyanonymous/ComfyUI",
            "X-Title": app_name
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, model)
    
    def _send_request_and_process(self, url, headers, request_data, model):
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"💬 正在生成文本... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                print(f"🌐 使用OpenRouter API: {url}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取响应文本
                    response_text = ""
                    model_info = ""
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"]
                    
                    # 提取模型使用信息
                    if "usage" in result:
                        usage = result["usage"]
                        model_info = f"模型: {model}\n"
                        model_info += f"输入tokens: {usage.get('prompt_tokens', 'N/A')}\n"
                        model_info += f"输出tokens: {usage.get('completion_tokens', 'N/A')}\n"
                        model_info += f"总tokens: {usage.get('total_tokens', 'N/A')}"
                    
                    # 如果有模型ID信息
                    if "model" in result:
                        model_info += f"\n实际使用模型: {result['model']}"
                    
                    if not response_text:
                        response_text = "请求已发送，但未收到有效响应"
                    
                    print(f"✅ 文本生成完成，响应长度: {len(response_text)} 字符")
                    return (response_text, model_info)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                        
                        # 提取具体错误信息
                        if "error" in error_detail:
                            error_msg = error_detail["error"].get("message", str(error_detail["error"]))
                        else:
                            error_msg = str(error_detail)
                            
                    except:
                        error_msg = response.text
                        print(f"❌ 错误文本: {error_msg}")
                    
                    if attempt == max_retries - 1:
                        raise ValueError(f"OpenRouter API错误 ({response.status_code}): {error_msg}")
                    
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


class OpenRouterImageEdit:
    """OpenRouter 图像编辑节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "编辑这张图像，让它更加美观", "multiline": True}),
                "model": ([
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "google/gemini-2.5-flash-image-preview",
                    "google/gemini-2.5-flash-image",
                ], {"default": "openai/gpt-4o-mini"}),
                "aspectRatio": ([
                    "auto",     # 自动选择最佳长宽比
                    "1:1",      # 正方形
                    "9:16",     # 竖屏
                    "16:9",     # 横屏
                    "3:4",      # 竖屏
                    "4:3",      # 横屏
                    "3:2",      # 横屏
                    "2:3",      # 竖屏
                    "5:4",      # 横屏
                    "4:5",      # 竖屏
                    "21:9",     # 超宽屏
                ], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 6664, "min": 1, "max": 6664}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "system_instruction": ("STRING", {"default": "", "multiline": True, "placeholder": "可选：系统提示词，为空时不发送"}),
                "site_url": ("STRING", {"default": "https://github.com/comfyanonymous/ComfyUI", "multiline": False}),
                "app_name": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "mirror_url": ("STRING", {"default": "https://openrouter.ai", "multiline": False, "placeholder": "镜像站地址，默认为OpenRouter官方"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "edit_description")
    FUNCTION = "edit_image"
    CATEGORY = "Nano"

    def edit_image(self, api_key, prompt, model, aspectRatio="auto", temperature=0.7, top_p=1.0, max_tokens=6664,
                   seed=0, images=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None,
                   system_instruction="", site_url="https://github.com/comfyanonymous/ComfyUI", app_name="ComfyUI", mirror_url="https://openrouter.ai"):
        """编辑图像"""
        
        # 检查API密钥
        if not validate_api_key(api_key):
            raise ValueError("请提供有效的OpenRouter API密钥")
        
        # 收集所有图像
        all_images = []
        
        # 处理批次图像（向后兼容）
        if images is not None:
            batch_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
            all_images.extend(batch_images)
            print(f"📥 从批次图像收到 {len(batch_images)} 张图像")
        
        # 处理6个独立的图像输入
        individual_images = [image_1, image_2, image_3, image_4, image_5, image_6]
        image_names = ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6"]
        
        for i, img in enumerate(individual_images):
            if img is not None:
                pil_image = tensor_to_pil(img)
                all_images.append(pil_image)
                print(f"📥 收到 {image_names[i]}: {pil_image.size}")
        
        if not all_images:
            raise ValueError("请至少提供一张图像")
        
        print(f"📥 总共收到 {len(all_images)} 张图像进行编辑")
        
        # 构建消息内容
        message_content = [{"type": "text", "text": prompt.strip()}]
        
        # 添加图像到消息中
        for i, pil_image in enumerate(all_images):
            resized_image = resize_image_for_api(pil_image, max_size=2048)
            image_base64 = image_to_base64(resized_image, format='JPEG')
            
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
            print(f"📎 添加第 {i+1} 张图像到编辑请求中")
        
        # 构建API URL - 使用可配置的镜像站地址
        # 确保URL格式正确，移除末尾的斜杠
        base_url = mirror_url.rstrip('/')
        url = f"{base_url}/api/v1/chat/completions"
        
        # 构建请求数据
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # 如果是支持图像生成的模型，添加modalities参数
        if any(gen_model in model for gen_model in ["gemini", "gpt-4o"]):
            request_data["modalities"] = ["image", "text"]
        
        # 只有当长宽比不是 "auto" 时才添加 imageConfig 到 generationConfig
        if aspectRatio != "auto":
            request_data["generationConfig"] = {
                "imageConfig": {
                    "aspectRatio": aspectRatio
                }
            }
        
        # 只有当系统提示词不为空时才添加 systemInstruction
        if system_instruction and system_instruction.strip():
            request_data["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction.strip()}
                ]
            }
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": app_name
        }
        
        # 发送请求并处理响应
        return self._send_edit_request_and_process(url, headers, request_data, model, all_images[0])
    
    def _send_edit_request_and_process(self, url, headers, request_data, model, fallback_image):
        """发送编辑请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"✏️ 正在编辑图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                print(f"🌐 使用OpenRouter API: {url}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取编辑结果
                    edited_image = None
                    edit_description = ""
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        
                        # 提取文本描述
                        if "content" in message:
                            edit_description = message["content"]
                        
                        # 检查是否有生成的图像
                        if "images" in message and message["images"]:
                            for image_item in message["images"]:
                                if "image_url" in image_item:
                                    image_url = image_item["image_url"]["url"]
                                    
                                    # 处理base64数据URL
                                    if image_url.startswith("data:image/"):
                                        try:
                                            base64_data = image_url.split(",")[1]
                                            pil_image = base64_to_image(base64_data)
                                            edited_image = pil_to_tensor(pil_image)
                                            print("✅ 成功获取编辑后的图像")
                                            break
                                        except Exception as e:
                                            print(f"⚠️ 解码编辑图像失败: {e}")
                                    
                                    # 处理URL
                                    elif image_url.startswith("http"):
                                        try:
                                            img_response = requests.get(image_url, timeout=30)
                                            img_response.raise_for_status()
                                            pil_image = Image.open(io.BytesIO(img_response.content))
                                            edited_image = pil_to_tensor(pil_image)
                                            print("✅ 成功从URL获取编辑图像")
                                            break
                                        except Exception as e:
                                            print(f"⚠️ 从URL获取编辑图像失败: {e}")
                    
                    # 如果没有编辑后的图像，返回原图像
                    if edited_image is None:
                        print("⚠️ 未收到编辑后的图像，返回原图像")
                        edited_image = pil_to_tensor(fallback_image)
                        if not edit_description:
                            edit_description = "图像编辑请求已处理，但未生成新图像"
                    
                    if not edit_description:
                        edit_description = "图像编辑完成"
                    
                    print(f"✅ 图像编辑完成，输出张量形状: {edited_image.shape}")
                    return (edited_image, edit_description)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                        
                        if "error" in error_detail:
                            error_msg = error_detail["error"].get("message", str(error_detail["error"]))
                        else:
                            error_msg = str(error_detail)
                            
                    except:
                        error_msg = response.text
                        print(f"❌ 错误文本: {error_msg}")
                    
                    if attempt == max_retries - 1:
                        raise ValueError(f"OpenRouter 图像编辑错误 ({response.status_code}): {error_msg}")
                    
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


class OpenRouterMultimodalImageGeneration:
    """OpenRouter 多模态图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "Generate a beautiful landscape painting", "multiline": True}),
                "model": ([
                    "google/gemini-2.5-flash-image",
                    "google/gemini-2.5-flash-image-preview",
                    "google/gemini-2.0-flash-preview-image-generation",
                    "openai/gpt-4o",
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3.5-sonnet",
                    "meta-llama/llama-3.2-90b-vision-instruct"
                ], {"default": "openai/gpt-4o-mini"}),
                "aspect_ratio": ([
                    "auto",     # 自动选择最佳长宽比
                    "1:1",      # 正方形
                    "9:16",     # 竖屏
                    "16:9",     # 横屏
                    "3:4",      # 竖屏
                    "4:3",      # 横屏
                    "3:2",      # 横屏
                    "2:3",      # 竖屏
                    "5:4",      # 横屏
                    "4:5",      # 竖屏
                    "21:9",     # 超宽屏
                ], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 6664, "min": 1, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "site_url": ("STRING", {"default": "", "multiline": False}),
                "app_name": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "system_instruction": ("STRING", {"default": "", "multiline": True, "placeholder": "可选：系统提示词，为空时不发送"}),
                "mirror_url": ("STRING", {"default": "https://openrouter.ai", "multiline": False, "placeholder": "镜像站地址，默认为OpenRouter官方"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response_text")
    FUNCTION = "generate_image"
    CATEGORY = "Nano"

    def generate_image(self, api_key, prompt, model, aspect_ratio="auto", temperature=1.0, top_p=0.95, max_output_tokens=6664,
                      seed=0, site_url="", app_name="ComfyUI", system_instruction="", mirror_url="https://openrouter.ai"):
        """使用多模态模型生成图像"""
        
        # 检查API密钥
        if not validate_api_key(api_key):
            raise ValueError("请提供有效的OpenRouter API密钥")
        
        # 构建API URL - 使用可配置的镜像站地址
        # 确保URL格式正确，移除末尾的斜杠
        base_url = mirror_url.rstrip('/')
        url = f"{base_url}/api/v1/chat/completions"
        
        # 构建请求数据 - 使用Gemini的多模态格式
        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt.strip()
                }
            ],
            "modalities": ["image", "text"],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens
        }
        
        # 只有当长宽比不是 "auto" 时才添加 imageConfig 到 generationConfig
        if aspect_ratio != "auto":
            request_data["generationConfig"] = {
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            }
        
        # 只有当系统提示词不为空时才添加 systemInstruction
        if system_instruction and system_instruction.strip():
            request_data["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction.strip()}
                ]
            }
        
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url if site_url else "https://github.com/comfyanonymous/ComfyUI",
            "X-Title": app_name
        }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url, headers, request_data, model)
    
    def _send_request_and_process(self, url, headers, request_data, model):
        """发送请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 180)  # 图像生成需要更长时间
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                print(f"🌐 使用OpenRouter API: {url}")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                # 成功响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"📋 API响应结构: {list(result.keys())}")
                    
                    # 提取生成的图像和文本
                    generated_image = None
                    response_text = ""
                    
                    if "choices" in result and result["choices"]:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        
                        # 提取文本响应
                        if "content" in message:
                            response_text = message["content"]
                        
                        # 提取图像
                        if "images" in message and message["images"]:
                            for image_item in message["images"]:
                                if "image_url" in image_item:
                                    image_url = image_item["image_url"]["url"]
                                    
                                    # 处理base64数据URL
                                    if image_url.startswith("data:image/"):
                                        try:
                                            # 提取base64数据
                                            base64_data = image_url.split(",")[1]
                                            pil_image = base64_to_image(base64_data)
                                            generated_image = pil_to_tensor(pil_image)
                                            print("✅ 成功从base64数据生成图像")
                                            break
                                        except Exception as e:
                                            print(f"⚠️ 解码base64图像失败: {e}")
                                    
                                    # 处理URL
                                    elif image_url.startswith("http"):
                                        try:
                                            img_response = requests.get(image_url, timeout=30)
                                            img_response.raise_for_status()
                                            pil_image = Image.open(io.BytesIO(img_response.content))
                                            generated_image = pil_to_tensor(pil_image)
                                            print("✅ 成功从URL获取图像")
                                            break
                                        except Exception as e:
                                            print(f"⚠️ 从URL获取图像失败: {e}")
                    
                    # 如果没有生成图像，创建一个默认图像
                    if generated_image is None:
                        print("⚠️ 未能获取生成的图像，创建默认图像")
                        default_image = Image.new('RGB', (512, 512), (128, 128, 128))
                        generated_image = pil_to_tensor(default_image)
                        if not response_text:
                            response_text = "图像生成请求已发送，但未收到图像数据"
                    
                    if not response_text:
                        response_text = "图像已生成"
                    
                    print(f"✅ 图像生成完成，输出张量形状: {generated_image.shape}")
                    return (generated_image, response_text)
                
                # 处理错误响应
                else:
                    print(f"❌ HTTP状态码: {response.status_code}")
                    try:
                        error_detail = response.json()
                        print(f"❌ 错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                        
                        # 提取具体错误信息
                        if "error" in error_detail:
                            error_msg = error_detail["error"].get("message", str(error_detail["error"]))
                        else:
                            error_msg = str(error_detail)
                            
                    except:
                        error_msg = response.text
                        print(f"❌ 错误文本: {error_msg}")
                    
                    if attempt == max_retries - 1:
                        raise ValueError(f"OpenRouter 图像生成错误 ({response.status_code}): {error_msg}")
                    
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
    "OpenRouterMirror": OpenRouterMirror,
    "OpenRouterImageEdit": OpenRouterImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterMirror": "OpenRouter 视觉分析",
    "OpenRouterImageEdit": "OpenRouter 图像编辑",
}
