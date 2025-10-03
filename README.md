# Gemini ComfyUI Plugin

一个强大的 ComfyUI 插件，集成 Google Gemini API 的图像生成和编辑功能。支持纯文本生成图像和基于现有图像的智能编辑。

## ✨ 功能特性

- 🎨 **图像生成**: 使用 Gemini 模型从文本提示生成高质量图像
- 🖼️ **图像编辑**: 基于输入图像和文本指令进行智能编辑
- 📐 **长宽比控制**: 支持多种图像尺寸比例 (1:1, 9:16, 16:9, 3:4, 4:3, 3:2, 2:3, 5:4, 4:5, 21:9)
- 🌐 **镜像站支持**: 支持自定义API地址，适配国内镜像站和代理服务
- 🤖 **多平台AI**: 支持OpenRouter统一接口，访问GPT-4、Claude、Llama等多种模型
- 👁️ **视觉分析**: 强大的图像理解和分析能力
- 🔄 **智能重试**: 内置限流处理和错误恢复机制
- 🚀 **多模型支持**: 支持最新的 Gemini 2.5 和 2.0 模型
- 🛡️ **稳定性**: 增强的错误处理和超时管理

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 获取 API Key

#### Google Gemini API Key
1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建新的 API Key
3. 在Gemini节点中输入你的 API Key

#### OpenRouter API Key
1. 访问 [OpenRouter](https://openrouter.ai/keys)
2. 注册账户并创建 API Key
3. 在OpenRouter节点中输入你的 API Key (格式: sk-or-v1-...)

#### 关于 site_url 参数
- **用途**: OpenRouter用于统计API使用来源和计费追踪
- **可选**: 不填写也能正常工作
- **建议**: 填写你的项目网站URL，有助于：
  - 获得更好的技术支持
  - 参与OpenRouter的开发者计划
  - 透明的使用统计
- **示例**: `https://github.com/your-username/your-project`

## 📦 节点说明

### Gemini 图片生成 (GeminiImageGeneration)

从文本提示生成全新图像。

**输入参数:**
- `api_key`: Google Gemini API 密钥
- `prompt`: 图像生成提示词
- `model`: 选择模型 (gemini-2.5-flash-image-preview 或 gemini-2.0-flash-preview-image-generation)
- `aspect_ratio`: 图像长宽比 (1:1, 9:16, 16:9, 3:4, 4:3, 3:2, 2:3, 5:4, 4:5, 21:9)
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `max_output_tokens`: 最大输出令牌数

**输出:**
- `image`: 生成的图像 (IMAGE 类型)
- `response_text`: API 响应文本

### Gemini 图片编辑 (GeminiImageEdit)

基于输入图像进行智能编辑，支持单图和多图批量处理。

**输入参数:**
- `api_key`: Google Gemini API 密钥
- `images`: 输入图像 (IMAGE 类型，支持批量)
- `prompt`: 编辑指令
- `model`: 选择模型
- `aspect_ratio`: 图像长宽比 (1:1, 9:16, 16:9, 3:4, 4:3, 3:2, 2:3, 5:4, 4:5, 21:9)
- `temperature`: 创造性控制
- `top_p`: 采样控制
- `max_output_tokens`: 最大输出令牌数
- `process_mode`: 处理模式
  - `each_image_separately`: 分别处理每张图片 (默认)
  - `all_images_combined`: 所有图片合并处理
  - `first_image_only`: 仅处理第一张图片

**输出:**
- `edited_image`: 编辑后的图像 (IMAGE 类型)
- `response_text`: API 响应文本

**处理模式说明:**
- **each_image_separately**: 对每张图片单独处理，适合个性化编辑
- **all_images_combined**: 将所有图片一起处理，适合保持风格一致性
- **first_image_only**: 只处理第一张图片，适合快速测试

### Gemini 镜像站图片生成 (GeminiMirrorImageGeneration)

支持自定义API地址的图像生成节点，适配国内镜像站。

**输入参数:**
- `api_url`: 自定义API地址 (如: https://ai.comfly.chat)
- `api_key`: API 密钥
- `prompt`: 图像生成提示词
- `model`: 选择模型
- `temperature`: 创造性控制
- `top_p`: 采样控制
- `max_output_tokens`: 最大输出令牌数

**输出:**
- `image`: 生成的图像 (IMAGE 类型)
- `response_text`: API 响应文本

### Gemini 镜像站图片编辑 (GeminiMirrorImageEdit)

支持自定义API地址的图像编辑节点，适配国内镜像站。

**输入参数:**
- `api_url`: 自定义API地址 (如: https://ai.comfly.chat)
- `api_key`: API 密钥
- `image`: 输入图像 (IMAGE 类型)
- `prompt`: 编辑指令
- `model`: 选择模型
- `temperature`: 创造性控制
- `top_p`: 采样控制
- `max_output_tokens`: 最大输出令牌数

**输出:**
- `edited_image`: 编辑后的图像 (IMAGE 类型)
- `response_text`: API 响应文本

### OpenRouter 视觉分析 (OpenRouterMirror)

使用OpenRouter统一接口访问多种AI模型进行图像分析和理解。

**输入参数:**
- `api_key`: OpenRouter API 密钥
- `prompt`: 分析指令或问题
- `model`: 选择模型 (支持GPT-4o、Claude-3.5、Gemini、Llama等)
- `max_tokens`: 最大输出令牌数 (1-8192)
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `images`: 输入图像 (可选，IMAGE 类型)
- `site_url`: 网站URL (可选，用于统计)
- `app_name`: 应用名称 (可选，默认ComfyUI)

**输出:**
- `response_text`: AI分析结果文本
- `model_info`: 模型使用信息和token统计

### OpenRouter 文本生成 (OpenRouterTextGeneration)

使用OpenRouter接口进行纯文本生成，支持多种先进AI模型。

**输入参数:**
- `api_key`: OpenRouter API 密钥
- `prompt`: 文本生成提示
- `model`: 选择模型 (GPT-4o、Claude、Llama、Mistral等)
- `max_tokens`: 最大输出令牌数 (1-8192)
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `system_prompt`: 系统提示 (可选)
- `site_url`: 网站URL (可选)
- `app_name`: 应用名称 (可选)

**输出:**
- `response_text`: 生成的文本内容
- `model_info`: 模型使用信息和token统计

### OpenRouter Gemini 图像生成 (OpenRouterGeminiImageGeneration)

使用OpenRouter接口调用Google Gemini模型进行图像生成，专注于Gemini的多模态能力。

**输入参数:**
- `api_key`: OpenRouter API 密钥
- `prompt`: 图像生成提示词
- `model`: 选择Gemini模型
  - `google/gemini-2.5-flash-image-preview` (推荐)
  - `google/gemini-2.5-flash-image-preview:free` (🆓 **免费模型**: 每天50张)
  - `google/gemini-2.0-flash-preview-image-generation`
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `max_output_tokens`: 最大输出令牌数 (1-32768)
- `seed`: 随机种子 (0为自动随机，其他值用于可重复生成)
- `site_url`: 网站URL (可选)
- `app_name`: 应用名称 (可选)

**输出:**
- `generated_image`: 生成的图像 (IMAGE 类型)
- `response_text`: Gemini的文本响应

**特色功能:**
- 专门针对Google Gemini模型优化
- 支持Gemini的多模态输出格式
- 同时返回图像和文本描述
- 使用OpenRouter的统一API接口
- 🆓 **免费模型**: 支持`google/gemini-2.5-flash-image-preview:free`，每天可免费生成50张图片
- 🎲 **随机种子控制**: 支持种子参数，可实现可重复的图像生成

### OpenRouter 图像编辑 (OpenRouterImageEdit)

使用OpenRouter接口对现有图像进行智能编辑和修改。

**输入参数:**
- `api_key`: OpenRouter API 密钥
- `images`: 输入图像 (IMAGE 类型，支持批量)
- `prompt`: 编辑指令 (如："让天空变成日落色彩"、"添加一只猫"等)
- `model`: 选择AI模型 (GPT-4o、GPT-4o-mini、Gemini-2.5-flash)
- `temperature`: 创造性控制 (0.0-2.0)
- `top_p`: 采样控制 (0.0-1.0)
- `max_tokens`: 最大输出令牌数 (1-8192，默认8192)
- `site_url`: 网站URL (可选，用于统计，默认ComfyUI GitHub)
- `app_name`: 应用名称 (可选，默认ComfyUI)

**输出:**
- `edited_image`: 编辑后的图像 (IMAGE 类型)
- `edit_description`: 编辑过程的文字描述

**编辑能力:**
- 🎨 **风格转换**: 改变图像的艺术风格
- 🌈 **色彩调整**: 修改颜色、亮度、对比度
- ➕ **内容添加**: 在图像中添加新元素
- ✂️ **内容移除**: 去除不需要的部分
- 🔄 **场景变换**: 改变背景或环境
- 📐 **构图优化**: 调整图像布局和比例

## 🔧 技术特性

### 长宽比控制

支持多种图像尺寸比例，满足不同使用场景：

- **1:1** - 正方形，适合社交媒体头像、产品展示
- **9:16** - 竖屏，适合手机壁纸、短视频
- **16:9** - 横屏，适合桌面壁纸、视频封面
- **3:4** - 竖屏，适合海报、宣传图
- **4:3** - 横屏，适合传统照片比例
- **3:2** - 横屏，适合摄影作品
- **2:3** - 竖屏，适合杂志封面
- **5:4** - 横屏，适合艺术画作
- **4:5** - 竖屏，适合Instagram帖子
- **21:9** - 超宽屏，适合电影海报、横幅

### 智能重试机制

- **指数退避**: 自动调整重试间隔
- **限流处理**: 针对 429 错误的特殊处理
- **错误分类**: 根据不同错误类型采用不同策略

### 图像处理

- **格式转换**: 自动处理 RGB/RGBA 转换
- **Base64 编码**: 高效的图像数据传输
- **质量优化**: 95% JPEG 质量保证

### 错误处理

- **详细日志**: 完整的操作过程记录
- **友好提示**: 清晰的错误信息和解决建议
- **优雅降级**: 失败时的备用方案

## 📋 依赖项

```
torch>=1.9.0          # 深度学习框架
numpy>=1.21.0         # 数值计算
Pillow>=8.0.0         # 图像处理
requests>=2.25.0      # HTTP 请求

# 可选依赖（仅当使用Google API时需要）
# google-genai>=0.3.0   # Google AI SDK
# google-cloud-aiplatform>=1.25.0
# google-auth>=2.0.0
# google-auth-oauthlib>=0.5.0
```

## 🛠️ 使用示例

### 基础图像生成

```python
# 在 ComfyUI 中添加 "Gemini 图片生成" 节点
# 设置参数:
api_key = "your_gemini_api_key"
prompt = "A serene mountain landscape at sunset with a lake reflection"
model = "gemini-2.5-flash-image-preview"
aspect_ratio = "16:9"  # 横屏比例，适合桌面壁纸
```

### 图像编辑

```python
# 在 ComfyUI 中添加 "Gemini 图片编辑" 节点
# 连接输入图像并设置参数:
prompt = "Add a rainbow in the sky"
aspect_ratio = "4:3"  # 传统照片比例
```

### 使用镜像站

```python
# 在 ComfyUI 中添加 "Gemini 镜像站图片生成" 节点
# 设置自定义API地址:
api_url = "https://ai.comfly.chat"  # 或其他镜像站地址
api_key = "your_api_key"
prompt = "A beautiful sunset over the ocean"
aspect_ratio = "21:9"  # 超宽屏比例，适合电影海报
```

### 使用OpenRouter进行图像分析

```python
# 在 ComfyUI 中添加 "OpenRouter 视觉分析" 节点
# 设置参数:
api_key = "sk-or-v1-your_openrouter_api_key"
prompt = "请详细描述这张图片中的内容，包括颜色、构图和情感表达"
model = "openai/gpt-4o"  # 或其他支持视觉的模型
```

### 使用OpenRouter进行文本生成

```python
# 在 ComfyUI 中添加 "OpenRouter 文本生成" 节点
# 设置参数:
api_key = "sk-or-v1-your_openrouter_api_key"
prompt = "写一首关于人工智能的诗"
model = "anthropic/claude-3.5-sonnet"
system_prompt = "你是一位富有创意的诗人"
```

### 使用OpenRouter进行Gemini图像生成

```python
# 在 ComfyUI 中添加 "OpenRouter 多模态图像生成" 节点
# 设置参数:
api_key = "sk-or-v1-your_openrouter_api_key"
prompt = "Generate a majestic dragon flying over a mystical forest at sunset, highly detailed, fantasy art style"
model = "google/gemini-2.5-flash-image-preview"  # 推荐的Gemini模型
aspect_ratio = "3:2"  # 摄影作品比例
temperature = 1.0
top_p = 0.95
max_output_tokens = 8192
```

### 使用OpenRouter进行图像编辑

```python
# 在 ComfyUI 中添加 "OpenRouter 图像编辑" 节点
# 连接输入图像并设置参数:
api_key = "sk-or-v1-your_openrouter_api_key"
prompt = "将这张图片的天空改成美丽的日落色彩，添加一些云朵"
model = "openai/gpt-4o-mini"  # 推荐用于图像编辑
temperature = 0.7
max_tokens = 8192
site_url = "https://your-project-website.com"  # 可选：你的项目网站
```

## 🌐 支持的镜像站

插件支持以下API格式的镜像站，**所有镜像站地址都可以在节点中自定义配置**：

### Gemini 原生格式镜像站
- `https://ai.comfly.chat` - ComflyAI 镜像站（默认）
- `https://api.openai-proxy.com` - 代理服务
- 其他兼容 Gemini API 格式的镜像服务

### OpenRouter 统一接口
- `https://openrouter.ai` - OpenRouter 官方API（默认）
- 支持多种AI模型：
  - **文本/视觉模型**:
    - **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
    - **Anthropic**: Claude-3.5-sonnet, Claude-3-haiku
    - **Google**: Gemini-pro-1.5, Gemini-flash-1.5
    - **Meta**: Llama-3.1-405b, Llama-3.1-70b, Llama-3.2-90b-vision
    - **Qwen**: Qwen-2-vl-72b, Qwen-2-72b
    - **Microsoft**: Phi-3.5-vision
    - **Mistral**: Mistral-7b-instruct
  - **图像生成模型**:
    - **Google Gemini**: gemini-2.5-flash-image-preview, gemini-2.5-flash-image-preview:free (🆓 每天50张), gemini-2.0-flash-preview-image-generation

### 🔧 镜像站配置说明

所有镜像站节点都支持自定义镜像站地址：

1. **ComflyGeminiMirror 节点**：
   - 默认地址：`https://ai.comfly.chat`
   - 参数名：`mirror_url`
   - 支持任何兼容 Gemini API 格式的镜像服务

2. **OpenRouter 系列节点**：
   - 默认地址：`https://openrouter.ai`
   - 参数名：`mirror_url`
   - 支持任何兼容 OpenRouter API 格式的镜像服务

3. **配置示例**：
   ```
   # 使用自定义镜像站
   mirror_url = "https://your-custom-mirror.com"
   
   # 使用官方服务
   mirror_url = "https://openrouter.ai"  # OpenRouter
   mirror_url = "https://ai.comfly.chat"  # Comfly
   ```

## ⚠️ 注意事项

1. **API 配额**: 注意 API 的使用限制
2. **网络连接**: 确保稳定的网络连接
3. **图像大小**: 大图像可能需要更长处理时间
4. **模型选择**: 不同模型有不同的特性和限制
5. **免费模型限制**: 
   - `google/gemini-2.5-flash-image-preview:free` 每天限制50张图片
   - 免费配额重置时间以UTC时间为准
6. **随机种子说明**:
   - 设置为 0 时自动生成随机种子
   - 相同种子+相同提示词可以生成相似的图像
   - 不同模型的种子效果可能不同

## 🔍 故障排除

### 常见问题

**API Key 错误**
- 确保 API Key 格式正确且有效
- 检查 API Key 权限设置

**429 限流错误**
- 等待更长时间再试
- 检查 API 配额设置
- 考虑升级 API 计划

**网络超时**
- 检查网络连接
- 增加超时时间设置

**免费模型配额用完**
- 等待次日UTC时间重置
- 切换到付费模型
- 检查OpenRouter账户余额

**图像生成失败**
- 确保提示词符合内容政策
- 检查图像尺寸参数
- 尝试不同的随机种子

## 💡 使用技巧和最佳实践

### 图像生成优化
- **提示词优化**: 使用清晰、具体的描述词
- **风格控制**: 可以在提示词中指定艺术风格，如"photorealistic", "anime style", "watercolor painting"
- **细节控制**: 使用修饰词如"highly detailed", "8K resolution", "professional photography"

### 种子使用策略
- **实验阶段**: 使用随机种子(0)探索不同可能性
- **微调阶段**: 固定种子值，仅调整提示词进行精细控制
- **批量生成**: 使用不同种子生成多个变体

### 模型选择建议
- **快速测试**: 使用免费模型进行初步实验
- **高质量输出**: 切换到付费模型获得更好效果
- **成本控制**: 先用免费模型确定满意的提示词，再用付费模型生成最终结果

### 工作流程建议
1. 使用文本生成节点完善提示词
2. 用免费图像生成模型测试效果
3. 调整参数和种子值
4. 切换到付费模型生成高质量图像

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests 来改进这个项目！

## 📄 许可证

本项目遵循开源许可证。使用前请确保遵守相关API的使用条款。

---

**注意**: 使用本插件需要有效的API密钥。请确保遵守相关的使用条款和限制。