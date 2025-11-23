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
                "prompt": ("STRING", {"default": "Describe these images and edit them", "multiline": True}),
                "model": (["gemini-2.5-flash-image", "gemini-2.0-flash-preview-image-generation", "gemini-3-pro-image-preview"], {"default": "gemini-2.5-flash-image"}),
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
                "image_size": (["1K", "2K", "4K"], {
                    "default": "4K",
                    "tooltip": "图像分辨率（仅适用于 gemini-3-pro-image-preview 模型）"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
            },
            "optional": {
                "images": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
                "system_instruction": ("STRING", {"default": "", "multiline": True, "placeholder": "可选：系统提示词，为空时不发送"}),
                "enable_google_search": ("BOOLEAN", {"default": False, "tooltip": "启用 Google Search 工具，允许模型进行联网搜索"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
    FUNCTION = "process_images"
    CATEGORY = "Nano"

    def process_images(self, api_key, prompt, model, aspectRatio="auto", image_size="4K", seed=0, temperature=1.0, top_p=0.95, max_output_tokens=8192, system_instruction="", 
                      images=None, image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None,
                      image_7=None, image_8=None, image_9=None, image_10=None, image_11=None, image_12=None, image_13=None, image_14=None, enable_google_search=False):
        """处理图像并返回编辑后的图像和响应文本"""
        
        # 检查API密钥
        if not api_key:
            raise ValueError("请提供有效的Gemini API密钥")
        
        # 收集所有图像
        pil_images = []
        
        # 处理批次图像（向后兼容）
        if images is not None:
            batch_images = [tensor_to_pil(images[i]) for i in range(images.shape[0])]
            pil_images.extend(batch_images)
            print(f"📥 从批次图像收到 {len(batch_images)} 张图像")
        
        # 处理14个独立的图像输入
        individual_images = [image_1, image_2, image_3, image_4, image_5, image_6, 
                            image_7, image_8, image_9, image_10, image_11, image_12, image_13, image_14]
        image_names = ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6",
                      "image_7", "image_8", "image_9", "image_10", "image_11", "image_12", "image_13", "image_14"]
        
        for i, img in enumerate(individual_images):
            if img is not None:
                pil_image = tensor_to_pil(img)
                pil_images.append(pil_image)
                print(f"📥 收到 {image_names[i]}: {pil_image.size}")
        
        if not pil_images:
            raise ValueError("请至少提供一张图像")
        
        print(f"📥 总共收到 {len(pil_images)} 张图像进行处理")
        
        # 检查图片数量限制（gemini-3-pro-image-preview 支持最多14张图片）
        if model == "gemini-3-pro-image-preview":
            if len(pil_images) > 14:
                print(f"⚠️ 警告: gemini-3-pro-image-preview 模型最多支持14张图片，当前有 {len(pil_images)} 张，将只处理前14张")
                pil_images = pil_images[:14]
            elif len(pil_images) == 14:
                print(f"✅ 使用 gemini-3-pro-image-preview 模型处理14张图片（最大支持数量）")
        
        # 使用优化的批量处理模式处理图像
        print(f"🔄 使用优化的批量处理模式处理图像")
        print(f"ℹ️ Received seed {seed}, but the Gemini API does not currently support a seed parameter for image editing.")
        print(f"📐 使用长宽比: {aspectRatio}")
        if model == "gemini-3-pro-image-preview":
            print(f"📏 使用图像分辨率: {image_size}")
        if enable_google_search:
            print(f"🔍 已启用 Google Search 工具")
        edited_tensor, response_text = self._process_combined_images(api_key, pil_images, prompt, model, aspectRatio, image_size, temperature, top_p, max_output_tokens, system_instruction, enable_google_search)
        
        return (edited_tensor, response_text)
    
    def _process_combined_images(self, api_key: str, pil_images: List[Image.Image], prompt: str, model: str, aspectRatio: str, image_size: str,                                                                                         
                                temperature: float, top_p: float, max_output_tokens: int, system_instruction: str = "", enable_google_search: bool = False) -> Tuple[torch.Tensor, str]:
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
        
        # 根据模型类型选择API端点和配置格式
        is_gemini_3_pro = (model == "gemini-3-pro-image-preview")
        
        # 构建API URL - gemini-3-pro-image-preview 使用 streamGenerateContent
        if is_gemini_3_pro:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # 构建请求数据 - 更新为匹配官方示例的格式
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseModalities": ["IMAGE", "TEXT"]
        }
        
        # 根据模型类型设置图像配置
        if is_gemini_3_pro:
            # gemini-3-pro-image-preview 同时支持 aspectRatio 和 image_size
            image_config = {
                "image_size": image_size
            }
            if aspectRatio != "auto":
                image_config["aspectRatio"] = aspectRatio
            generation_config["imageConfig"] = image_config
        else:
            # 其他模型使用 aspectRatio
            if aspectRatio != "auto":
                generation_config["imageConfig"] = {
                    "aspectRatio": aspectRatio
                }
        
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": generation_config
        }
        
        # 添加系统提示词（如果提供）
        if system_instruction and system_instruction.strip():
            request_data["systemInstruction"] = {
                "parts": [{"text": system_instruction.strip()}]
            }
        
        # 添加 Google Search 工具（如果启用）
        if enable_google_search:
            request_data["tools"] = [
                {
                    "googleSearch": {}
                }
            ]
        
        # 设置请求头 - gemini-3-pro-image-preview 使用 key 参数而不是 x-goog-api-key
        if is_gemini_3_pro:
            url_with_key = f"{url}?key={api_key.strip()}"
            headers = {
                "Content-Type": "application/json"
            }
        else:
            url_with_key = url
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key.strip()
            }
        
        # 发送请求并处理响应
        return self._send_request_and_process(url_with_key, headers, request_data, pil_images[0], model, is_gemini_3_pro)
    
    def _parse_stream_response(self, response):
        """解析流式响应（streamGenerateContent）"""
        # 流式响应通常是 JSON Lines 格式，每行一个 JSON 对象
        # 但根据 API 文档，也可能返回单个 JSON 对象或数组
        try:
            # 尝试解析为单个 JSON
            result = response.json()
            
            # 如果结果是数组，取第一个元素
            if isinstance(result, list):
                print(f"📦 解析为JSON数组，长度: {len(result)}")
                if result:
                    result = result[0]
                    print(f"📦 使用数组第一个元素，keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                else:
                    print(f"⚠️ JSON数组为空")
                    return {"candidates": []}
            else:
                print(f"📦 解析为单个JSON对象，keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            return result
        except:
            # 如果是流式响应，解析每一行
            lines = response.text.strip().split('\n')
            print(f"📦 解析流式响应，共 {len(lines)} 行")
            final_result = {"candidates": []}
            
            for line_idx, line in enumerate(lines):
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        
                        # 如果chunk是数组，取第一个元素
                        if isinstance(chunk, list):
                            if chunk:
                                chunk = chunk[0]
                            else:
                                continue
                        
                        print(f"📦 第 {line_idx + 1} 行: keys={list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")
                        
                        # 合并流式响应的数据
                        if "candidates" in chunk:
                            if not final_result["candidates"]:
                                final_result["candidates"] = chunk["candidates"]
                                print(f"📦 初始化 candidates，数量: {len(final_result['candidates'])}")
                            else:
                                # 合并候选内容
                                for i, candidate in enumerate(chunk["candidates"]):
                                    if i < len(final_result["candidates"]):
                                        if "content" in candidate and "parts" in candidate["content"]:
                                            if "content" not in final_result["candidates"][i]:
                                                final_result["candidates"][i]["content"] = {"parts": []}
                                            parts_count = len(candidate["content"]["parts"])
                                            final_result["candidates"][i]["content"]["parts"].extend(candidate["content"]["parts"])
                                            print(f"📦 合并 candidate[{i}]，添加 {parts_count} 个 parts")
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第 {line_idx + 1} 行JSON解析失败: {e}")
                        continue
            
            print(f"📦 最终结果: candidates数量={len(final_result.get('candidates', []))}")
            return final_result if final_result["candidates"] else {"candidates": []}
    
    def _send_request_and_process(self, url: str, headers: dict, request_data: dict, 
                                 fallback_image: Image.Image, model: str, is_stream: bool = False) -> Tuple[torch.Tensor, str]:
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
                    # 处理流式响应（gemini-3-pro-image-preview）
                    if is_stream:
                        result = self._parse_stream_response(response)
                    else:
                        result = response.json()
                        # 如果结果是数组，取第一个元素
                        if isinstance(result, list):
                            print(f"📋 响应是数组，长度: {len(result)}")
                            if result:
                                result = result[0]
                            else:
                                result = {"candidates": []}
                    
                    print(f"📋 API响应结构: {list(result.keys()) if isinstance(result, dict) else type(result).__name__}")
                    
                    # 提取文本响应和编辑后的图片
                    response_text = ""
                    edited_image = None
                    
                    if isinstance(result, dict) and "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        print(f"📋 Candidate结构: {list(candidate.keys()) if isinstance(candidate, dict) else 'N/A'}")
                        
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            print(f"📋 找到 {len(parts)} 个 parts")
                            
                            for part_idx, part in enumerate(parts):
                                print(f"📋 Part[{part_idx}] keys: {list(part.keys()) if isinstance(part, dict) else 'N/A'}")
                                
                                # 提取文本
                                if "text" in part:
                                    text_content = part["text"]
                                    response_text += text_content
                                    print(f"📋 Part[{part_idx}] 包含文本，长度: {len(text_content)}")
                                
                                # 提取编辑后的图片 - 检查多种可能的字段名
                                inline_data = None
                                if "inline_data" in part:
                                    inline_data = part["inline_data"]
                                    print(f"📋 Part[{part_idx}] 包含 inline_data")
                                elif "inlineData" in part:
                                    inline_data = part["inlineData"]
                                    print(f"📋 Part[{part_idx}] 包含 inlineData")
                                
                                if inline_data:
                                    print(f"📋 inline_data keys: {list(inline_data.keys()) if isinstance(inline_data, dict) else 'N/A'}")
                                    if "data" in inline_data:
                                        try:
                                            image_data = inline_data["data"]
                                            mime_type = inline_data.get("mimeType", "unknown")
                                            print(f"📋 找到图片数据，mimeType: {mime_type}, 数据长度: {len(image_data)}")
                                            image_bytes = base64.b64decode(image_data)
                                            edited_image = Image.open(io.BytesIO(image_bytes))
                                            print(f"✅ 成功提取编辑后的图片，尺寸: {edited_image.size}")
                                        except Exception as e:
                                            print(f"⚠️ 解码图片失败: {e}")
                                            import traceback
                                            print(f"⚠️ 错误详情: {traceback.format_exc()}")
                        else:
                            print(f"⚠️ Candidate中没有content.parts，candidate keys: {list(candidate.keys())}")
                            # 尝试直接检查candidate中是否有图片数据
                            if "inline_data" in candidate or "inlineData" in candidate:
                                inline_data = candidate.get("inline_data") or candidate.get("inlineData")
                                if inline_data and "data" in inline_data:
                                    try:
                                        image_data = inline_data["data"]
                                        image_bytes = base64.b64decode(image_data)
                                        edited_image = Image.open(io.BytesIO(image_bytes))
                                        print("✅ 成功从candidate直接提取编辑后的图片")
                                    except Exception as e:
                                        print(f"⚠️ 从candidate解码图片失败: {e}")
                    else:
                        print(f"⚠️ 响应中没有candidates，result类型: {type(result).__name__}")
                        if isinstance(result, dict):
                            print(f"⚠️ result keys: {list(result.keys())}")
                        elif isinstance(result, list):
                            print(f"⚠️ result是数组，长度: {len(result)}")
                            if result:
                                print(f"⚠️ 数组第一个元素类型: {type(result[0]).__name__}")
                                if isinstance(result[0], dict):
                                    print(f"⚠️ 数组第一个元素keys: {list(result[0].keys())}")
                        # 打印完整的响应结构用于调试（限制长度）
                        try:
                            debug_str = json.dumps(result, indent=2, ensure_ascii=False)[:1000]
                            print(f"📋 响应内容预览: {debug_str}...")
                        except:
                            print(f"📋 无法序列化响应内容")
                    
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
    
    def _parse_stream_response_generate(self, response):
        """解析生成节点的流式响应（streamGenerateContent）"""
        # 流式响应通常是 JSON Lines 格式，每行一个 JSON 对象
        # 但根据 API 文档，也可能返回单个 JSON 对象或数组
        try:
            # 尝试解析为单个 JSON
            result = response.json()
            
            # 如果结果是数组，取第一个元素
            if isinstance(result, list):
                if result:
                    result = result[0]
                else:
                    return {"candidates": []}
            
            return result
        except:
            # 如果是流式响应，解析每一行
            lines = response.text.strip().split('\n')
            final_result = {"candidates": []}
            
            for line in lines:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        
                        # 如果chunk是数组，取第一个元素
                        if isinstance(chunk, list):
                            if chunk:
                                chunk = chunk[0]
                            else:
                                continue
                        
                        # 合并流式响应的数据
                        if "candidates" in chunk:
                            if not final_result["candidates"]:
                                final_result["candidates"] = chunk["candidates"]
                            else:
                                # 合并候选内容
                                for i, candidate in enumerate(chunk["candidates"]):
                                    if i < len(final_result["candidates"]):
                                        if "content" in candidate and "parts" in candidate["content"]:
                                            if "content" not in final_result["candidates"][i]:
                                                final_result["candidates"][i]["content"] = {"parts": []}
                                            final_result["candidates"][i]["content"]["parts"].extend(candidate["content"]["parts"])
                    except json.JSONDecodeError:
                        continue
            
            return final_result if final_result["candidates"] else {"candidates": []}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme", "multiline": True}),
                "model": (["gemini-2.5-flash-image", "gemini-2.0-flash-preview-image-generation", "gemini-3-pro-image-preview"], {"default": "gemini-2.5-flash-image"}),
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
                "image_size": (["1K", "2K", "4K"], {
                    "default": "4K",
                    "tooltip": "图像分辨率（仅适用于 gemini-3-pro-image-preview 模型）"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
            },
            "optional": {
                "system_instruction": ("STRING", {"default": "", "multiline": True, "placeholder": "可选：系统提示词，为空时不发送"}),
                "enable_google_search": ("BOOLEAN", {"default": False, "tooltip": "启用 Google Search 工具，允许模型进行联网搜索"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response_text")
    FUNCTION = "generate_images"
    CATEGORY = "Nano"

    def generate_images(self, api_key: str, prompt: str, model: str, aspectRatio: str, image_size: str, seed: int,
                        temperature: float, top_p: float, max_output_tokens: int, system_instruction: str = "", enable_google_search: bool = False) -> Tuple[torch.Tensor, str]:
        
        if not validate_api_key(api_key):
            raise ValueError("API Key格式无效或为空")
        
        if not prompt.strip():
            raise ValueError("提示词不能为空")

        print(f"ℹ️ Received seed {seed}, but the Gemini API does not currently support a seed parameter for image generation.")
        print(f"📐 使用长宽比: {aspectRatio}")
        if model == "gemini-3-pro-image-preview":
            print(f"📏 使用图像分辨率: {image_size}")
        if enable_google_search:
            print(f"🔍 已启用 Google Search 工具")

        # 根据模型类型选择API端点
        is_gemini_3_pro = (model == "gemini-3-pro-image-preview")
        
        if is_gemini_3_pro:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        generation_config = {
            "candidateCount": 1,
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "responseModalities": ["IMAGE", "TEXT"]
        }
        
        # 根据模型类型设置图像配置
        if is_gemini_3_pro:
            # gemini-3-pro-image-preview 同时支持 aspectRatio 和 image_size
            image_config = {
                "image_size": image_size
            }
            if aspectRatio != "auto":
                image_config["aspectRatio"] = aspectRatio
            generation_config["imageConfig"] = image_config
        else:
            # 其他模型使用 aspectRatio
            if aspectRatio != "auto":
                generation_config["imageConfig"] = {
                    "aspectRatio": aspectRatio
                }
        
        request_data = {
            "contents": [{
                "parts": [
                    {"text": prompt.strip()}
                ]
            }],
            "generationConfig": generation_config
        }
        
        # 只有当系统提示词不为空时才添加 systemInstruction
        if system_instruction and system_instruction.strip():
            request_data["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction.strip()}
                ]
            }
        
        # 添加 Google Search 工具（如果启用）
        if enable_google_search:
            request_data["tools"] = [
                {
                    "googleSearch": {}
                }
            ]
        
        # 根据模型类型设置请求头和URL
        if is_gemini_3_pro:
            url_with_key = f"{url}?key={api_key.strip()}"
            headers = {
                "Content-Type": "application/json"
            }
        else:
            url_with_key = url
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key.strip()
            }
        
        return self._send_request_and_generate_images(url_with_key, headers, request_data, model, is_gemini_3_pro)

    def _send_request_and_generate_images(self, url: str, headers: dict, request_data: dict, model: str, is_stream: bool = False) -> Tuple[torch.Tensor, str]:
        """发送生成请求并处理响应"""
        
        max_retries = 5
        timeout = DEFAULT_CONFIG.get("timeout", 120)
        
        for attempt in range(max_retries):
            try:
                print(f"🎨 正在生成图像... (尝试 {attempt + 1}/{max_retries}) 使用模型: {model}")
                
                response = requests.post(url, headers=headers, json=request_data, timeout=timeout)
                
                if response.status_code == 200:
                    # 处理流式响应（gemini-3-pro-image-preview）
                    if is_stream:
                        result = self._parse_stream_response_generate(response)
                    else:
                        result = response.json()
                        # 如果结果是数组，取第一个元素
                        if isinstance(result, list):
                            print(f"📋 响应是数组，长度: {len(result)}")
                            if result:
                                result = result[0]
                            else:
                                result = {"candidates": []}
                    generated_images = []
                    response_texts = []
                    
                    if isinstance(result, dict) and "candidates" in result and result["candidates"]:
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