# STAGE 10 IMPLEMENTATION DECISIONS

## 1. What is known
The following values are established as `PAPER-FACT` from the ACHG-CLIP paper:
- **CIFAR-100**: 100 total classes, 60 base, 40 incremental, 8 incremental sessions (5 classes/session), 5 shots/class, 32x32 native resolution.
- **miniImageNet**: 100 total classes, 60 base, 40 incremental, 8 incremental sessions (5 classes/session), 5 shots/class, 84x84 native resolution.
- **CUB-200-2011**: 200 total classes, 100 base, 100 incremental, 10 incremental sessions (10 classes/session), 5 shots/class, 224x224 native resolution.
- **Training protocol**: Base session uses batch size 4 and 3 epochs. Incremental sessions use batch size 4 and 5 epochs.

## 2. What is unknown
The following values are completely absent from the paper and from its verifiable references, and the official repository is a `404 Not Found`:
- Exact class identities and ordering (e.g. which 60 classes form the base for CIFAR-100).
- Explicit data preprocessing pipelines (resize logic, normalization, augmentations).
- Exact train/val/test data splits beyond the standard FSCIL testing formulation.

## 3. What we are substituting
Because the exact splits and preprocessing are unrecoverable, we must adopt a functional reproduction track. 

| Dataset | Component | Paper Fact | Selected Value | Source | Provenance | Confidence | Reproduction Risk |
|---|---|---|---|---|---|---|---|
| All | Class Ordering | NO | Deterministic Pseudo-Random | Functional Baseline | IMPLEMENTATION-CHOICE | High (for functionality) | High (for exact numeric reproduction) |
| All | Base/Novel Split | NO | Determined by index from ordering | Functional Baseline | IMPLEMENTATION-CHOICE | High | High |
| All | Few-Shot Sampling | NO | Deterministic pseudo-random sample of available training data | Functional Baseline | IMPLEMENTATION-CHOICE | High | Medium |
| All | Preprocessing | NO | Resize(224) + CenterCrop(224) | Standard CLIP practice | IMPLEMENTATION-CHOICE | Medium | High |
| All | Normalization | NO | Mean `[0.481, 0.457, 0.408]`, Std `[0.268, 0.261, 0.275]` | Standard CLIP practice | IMPLEMENTATION-CHOICE | High (CLIP requires this) | Low (assuming standard CLIP) |

## 4. Why each substitution was necessary
- **Class Ordering and Split**: Without the exact permutation used by the authors or a cited reference like CEC (Tao et al.), the only way to build the Base and Incremental sessions is to generate a reproducible pseudo-random split. This conceptually mirrors CEC/FACT but doesn't pretend to be the exact unrecoverable list.
- **Preprocessing (Resize 224)**: While the paper lists CIFAR-100 as 32x32, standard CLIP ViT backbones expect 224x224 inputs. Feeding 32x32 directly to a ViT-B/32 yields 1 patch, severely breaking internal sequences. We substitute 224x224 resizing to ensure the frozen CLIP architecture functions.
- **Normalization**: Pre-trained CLIP models are highly sensitive to normalization. Since the authors froze the CLIP backbone, they overwhelmingly likely used standard CLIP normalization.

## 5. How to replace the substitutions later if author code appears
- **Class Ordering**: If the authors provide a list of class indices, it can be passed into the `FSCILDataManager` directly, overriding the pseudo-random generation.
- **Preprocessing**: The `transforms.py` module isolates these choices. If the authors specify different crop logic or augmentation, that module can be updated without touching the dataloader or session logic.
