# CLIP Query Encoder (with Few-Shot Adapter)

## What is this?
**CLIPQueryEncoder** wraps CLIP ViT-B/32 and (optionally) a few-shot adapter.  
It outputs a **512-D text embedding**.  

## Files
- `clip_query_encoder.py` — encoder + optional adapter wrapper  
- `few_shot_adapter.py` — trains the Linear adapter on `prompts_train.yaml`, evaluates on `prompts_test.yaml`  
- `prompt_dataset.py` — YAML loader  
- `prompts_train.yaml`, `prompts_test.yaml` — text prompts  
- `adapter/adapter_linear.pt`, `adapter/classes.json` — saved adapter  

## How to load for inference
```python
from query_encoder.clip_query_encoder import CLIPQueryEncoder

enc = CLIPQueryEncoder(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    adapter_path="query_encoder/adapter/adapter_linear.pt",  
    classes_path="query_encoder/adapter/classes.json",       
)

# 512-D embedding for cross-attention (adapter-projected if available)
query_vec = enc.get_embedding(["referee shows a red card"], adapted=True)  # shape [1,512]

# (optional) class probabilities (for logging/monitoring)
top3 = enc.topk("referee shows a red card", k=3)
