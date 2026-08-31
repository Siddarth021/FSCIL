# STAGE 10 DATASET PROTOCOL INVESTIGATION

## 1. Reference [56] identification
- **Reference Number:** [56]
- **Title:** "Learning multiple layers of features from tiny images"
- **Authors:** A. Krizhevsky and G. Hinton
- **Year:** 2009
- **Institution:** Univ. Toronto, Tech. Rep. UTML TR 2009-007
- **Usage in Target Paper:** Cited as the primary source for the CIFAR-100 dataset.

## 2. Reference [57] identification
- **Reference Number:** [57]
- **Title:** "ImageNet large scale visual recognition challenge"
- **Authors:** O. Russakovsky et al.
- **Year:** 2015
- **Journal:** Int. J. Comput. Vis., vol. 115, no. 3, pp. 211–252
- **Usage in Target Paper:** Cited as the primary source for the miniImageNet dataset.

## 3. Reference [58] identification
- **Reference Number:** [58]
- **Title:** "The Caltech-UCSD Birds-200-2011 Dataset"
- **Authors:** C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie
- **Year:** 2011
- **Institution:** California Inst. Technol., Tech. Rep. CNS-TR-2011-001
- **Usage in Target Paper:** Cited as the primary source for the CUB-200-2011 dataset.

## 4. CIFAR-100 findings
The target paper defines 60 base classes and 8 incremental sessions (40 classes, 5 classes/session, 5-shot). Reference [56] only introduces the base dataset and does not contain any FSCIL splits.

## 5. miniImageNet findings
The target paper defines 60 base classes and 8 incremental sessions (40 classes, 5 classes/session, 5-shot). Reference [57] introduces the ImageNet challenge and does not contain any FSCIL splits.

## 6. CUB-200-2011 findings
The target paper defines 100 base classes and 10 incremental sessions (100 classes, 10 classes/session, 5-shot). Reference [58] introduces the dataset and does not contain any FSCIL splits.

## 7. Exact class split findings
- **Base Session & Incremental Sessions:** The exact identities of the classes allocated to the base session and the subsequent incremental sessions are **UNRESOLVED**. 
- **Split File/Seed:** No random seed, permutation, or specific FSCIL protocol benchmark (like CEC or FACT) is explicitly cited as the source of the splits.

## 8. Session protocol findings
- **CIFAR-100/miniImageNet:** 8 sessions (Session 1-8: 5 novel classes, 5 shots).
- **CUB-200-2011:** 10 sessions (Session 1-10: 10 novel classes, 5 shots).
- **Train/Validation/Test:** The paper only specifies cumulative evaluation on all classes seen so far. No explicit validation split protocol is specified.

## 9. Preprocessing findings
- **CIFAR-100:** Resize, crop, flip, and normalization are **UNRESOLVED**. (Image resolution stated as 32x32 is a PAPER-FACT).
- **miniImageNet:** Resize, crop, flip, and normalization are **UNRESOLVED**. (Image resolution stated as 84x84 is a PAPER-FACT).
- **CUB-200-2011:** Resize, crop, flip, and normalization are **UNRESOLVED**. (Image resolution stated as 224x224 is a PAPER-FACT).

## 10. Author-code findings
- The target paper explicitly points to a GitHub repository: `https://github.com/aries-yqian/ACHG-CLIP`.
- A direct clone request (`git clone https://github.com/aries-yqian/ACHG-CLIP.git`) returns a `404 Not Found` error. The repository is officially non-functional/private/deleted.

## 11. Source hierarchy
1. Target Paper (ACHG-CLIP)
2. References [56], [57], [58]
3. Standard FSCIL protocols (CEC, FACT, TEEN)

## 12. Conflicts
There are no direct conflicts between the paper and references [56], [57], [58]. The paper simply fails to cite an FSCIL benchmark protocol paper (e.g., Tao et al. 2020) and instead cites the original 2009-2015 dataset creation papers, which do not contain FSCIL splits. The official repository being a 404 creates an unresolvable vacuum of information.

## 13. Resolved facts
- **PAPER-FACT:** Dataset resolutions, number of total classes, number of base classes, number of incremental classes, number of incremental sessions, classes per session, and shots per class.

## 14. Remaining unresolved facts
- Exact class identities (permutations) for base and incremental sessions.
- Detailed train/val/test splits beyond cumulative accuracy evaluations.
- Image preprocessing transforms (resize, crop, normalization, augmentations).

## 15. Reproduction-risk assessment
The reproduction risk is **HIGH**. Without the exact class permutations and specific preprocessing pipelines, achieving the precise baseline accuracy targets (e.g. 82.30% for CIFAR-100) is highly unlikely due to the sensitivity of few-shot models to class orderings and data distributions. 

## 16. Recommended implementation decisions
Since the exact class splits and preprocessing are strictly unrecoverable due to the 404 on the official repo and lack of citation in the paper:
- **Dataset splits:** Adopt the standard CEC benchmark splits for CIFAR-100, miniImageNet, and CUB-200. This is the defacto standard for FSCIL.
- **Preprocessing:** Use standard CLIP normalization (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) with resizing appropriately matched to the backbone model's expectation.
- These must be formally recorded as **IMPLEMENTATION-CHOICE** in code comments and logs, never as PAPER-FACT.
