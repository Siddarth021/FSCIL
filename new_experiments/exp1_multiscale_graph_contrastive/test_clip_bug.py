import torch
import transformers

processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=["a photo of a bird"], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
input_ids = inputs.input_ids
print("Original EOS index:", input_ids.argmax(dim=-1).item())
