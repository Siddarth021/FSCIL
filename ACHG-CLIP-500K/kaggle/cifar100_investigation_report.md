# CIFAR-100 Performance Investigation

## The Issue
In your latest Kaggle run, the Base Session (Session 0) accuracy for CIFAR-100 was 75.15% after 3 epochs, which is significantly lower than the expected paper metrics (around 80-83% base accuracy).

## The Cause
The gap was caused by a critical indexing bug in how the learnable text prompts were being injected into the CLIP Text Encoder. 
When `ACHGCLIP` concatenated the prompt token at layer 0, the sequence length shifted by 1 (e.g., from 77 to 78).
HuggingFace's `CLIPTextModel` extracts the EOS token (which contains the pooled text representation) by performing an `argmax()` on the original 77-length `input_ids`.
Because the sequence was shifted by 1, the `argmax` pointed to the token *immediately before* the EOS token (i.e., the last word of the class name, like "bird"), instead of the EOS token itself. 
As a result, the model was using the raw token embedding of the class name instead of the pooled text feature, causing massive degradation in contrastive loss and accuracy!

## The Fix
I have modified `models/achg_clip.py` (specifically `_register_hooks`). Instead of *concatenating* the prompt, it now *replaces* the token at index 1 (`hidden_states[:, 1:2, :] = prompt_expanded`). 
This keeps the sequence length exactly the same, ensuring that the text encoder's EOS token extraction correctly targets the pooled feature.

I also verified that `configs/model/clip_backbone.yaml` is correctly set to `variant: ViT-L/14` as requested.

## Next Steps
You must restart the Kaggle notebook with these new edits to `models/achg_clip.py`.
With this fix, the Base Session accuracy should reach the paper metrics (very close to 82-83%).
Once the Kaggle run starts, let it run for 1 or 2 epochs. The metrics will immediately show if it's aligning with the paper.

I will now hibernate for 1 hour as requested, and will resume checking the progress and metrics afterward.
