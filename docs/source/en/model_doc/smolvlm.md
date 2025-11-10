<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-02-20 and added to Hugging Face Transformers on 2025-02-20.*

# SmolVLM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[SmolVLM2](https://huggingface.co/papers/2504.05299) ([blog post](https://huggingface.co/blog/smolvlm2)) is an adaptation of the Idefics3 model with two main differences:

- It uses SmolLM2 for the text model.
- It supports multi-image and video inputs

## Usage tips

Input images are processed either by upsampling (if resizing is enabled) or at their original resolution. The resizing behavior depends on two parameters: do_resize and size.

Videos should not be upsampled.

If `do_resize` is set to `True`, the model resizes images so that the longest edge is 4*512 pixels by default.
The default resizing behavior can be customized by passing a dictionary to the `size` parameter. For example, `{"longest_edge": 4 * 512}` is the default, but you can change it to a different value if needed.

Here's how to control resizing and set a custom size:

```python
image_processor = SmolVLMImageProcessor(do_resize=True, size={"longest_edge": 2 * 512}, max_image_size=512)
```

Additionally, the `max_image_size` parameter, which controls the size of each square patch the image is decomposed into, is set to 512 by default but can be adjusted as needed. After resizing (if applicable), the image processor decomposes the images into square patches based on the `max_image_size` parameter.

This model was contributed by [orrzohar](https://huggingface.co/orrzohar).

## Usage example

### Single Media inference

The model can accept both images and videos as input, but you should use only one of the modalities at a time. Here's an example code for that.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

conversation = [
    {
        "role": "user",
        "content":[
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)


# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

### Batch Mixed Media Inference

The model can batch inputs composed of several images/videos and text. Here is an example.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {"role": "user","content": "who are you?"}
]


conversations = [conversation1, conversation2, conversation3]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

### Error Handling and Best Practices

When working with SmolVLM, it's important to handle common errors gracefully. Here are examples of robust error handling for typical use cases:

#### Handling File Path Errors

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path
from PIL import Image

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

def process_image_safely(image_path: str, prompt: str):
    """Process an image with proper error handling.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        
    Returns:
        Generated text description or None if processing fails
    """
    # Validate file exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Validate file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    if Path(image_path).suffix.lower() not in valid_extensions:
        print(f"Error: Unsupported image format. Supported formats: {valid_extensions}")
        return None
    
    # Validate image can be opened
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        print(f"Error: Cannot open image file. It may be corrupted: {e}")
        return None
    
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return generated_text
        
    except torch.cuda.OutOfMemoryError:
        print("Error: Out of GPU memory. Try reducing image resolution or using a smaller model.")
        return None
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None

# Example usage
result = process_image_safely("path/to/your/image.jpg", "Describe this image.")
if result:
    print(f"Description: {result}")
```

#### Memory Management for Large Batches

When processing multiple images or videos, memory management becomes critical:

```python
import torch
import gc

def process_images_batch(image_paths, prompts, batch_size=1):
    """Process multiple images with memory-efficient batching.
    
    Args:
        image_paths: List of paths to image files
        prompts: List of prompts (one per image)
        batch_size: Number of images to process at once (default: 1)
        
    Returns:
        List of generated descriptions
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        
        # Clear GPU cache before each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            conversations = []
            for path, prompt in zip(batch_paths, batch_prompts):
                if not Path(path).exists():
                    print(f"Skipping missing file: {path}")
                    results.append(None)
                    continue
                    
                conversations.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": path},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ])
            
            if not conversations:
                continue
                
            inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            
            output_ids = model.generate(**inputs, max_new_tokens=128)
            generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(generated_texts)
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM error at batch starting at index {i}. Try reducing batch_size.")
            # Process images one by one as fallback
            for path, prompt in zip(batch_paths, batch_prompts):
                result = process_image_safely(path, prompt)
                results.append(result)
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

# Example: Process multiple images with memory management
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
prompts = ["Describe this image"] * len(image_paths)
descriptions = process_images_batch(image_paths, prompts, batch_size=2)
```

#### Handling Video Processing Errors

```python
import cv2

def validate_video(video_path: str):
    """Validate video file before processing.
    
    Args:
        video_path: Path to video file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not Path(video_path).exists():
        return False, f"Video file not found: {video_path}"
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    if Path(video_path).suffix.lower() not in valid_extensions:
        return False, f"Unsupported video format. Supported: {valid_extensions}"
    
    # Try to open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file. It may be corrupted."
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return False, "Video has no frames."
        
        cap.release()
        return True, None
        
    except Exception as e:
        return False, f"Error reading video: {e}"

def process_video_safely(video_path: str, prompt: str):
    """Process video with error handling."""
    # Validate video first
    is_valid, error_msg = validate_video(video_path)
    if not is_valid:
        print(f"Error: {error_msg}")
        return None
    
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

# Example usage
result = process_video_safely("path/to/video.mp4", "Describe this video in detail")
if result:
    print(result)
```

These error handling patterns help ensure your SmolVLM applications are robust and provide helpful feedback when issues occur.

## SmolVLMConfig

[[autodoc]] SmolVLMConfig

## SmolVLMVisionConfig

[[autodoc]] SmolVLMVisionConfig

## Idefics3VisionTransformer

[[autodoc]] SmolVLMVisionTransformer

## SmolVLMModel

[[autodoc]] SmolVLMModel
    - forward

## SmolVLMForConditionalGeneration

[[autodoc]] SmolVLMForConditionalGeneration
    - forward

## SmolVLMImageProcessor

[[autodoc]] SmolVLMImageProcessor
    - preprocess

## SmolVLMImageProcessorFast

[[autodoc]] SmolVLMImageProcessorFast
    - preprocess

## SmolVLMVideoProcessor

[[autodoc]] SmolVLMVideoProcessor
    - preprocess

## SmolVLMProcessor

[[autodoc]] SmolVLMProcessor
    - __call__
