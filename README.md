# Multi-Label-Text-Classification-DB-Loss
Multi-label Text Classification using different Loss Functions to handle Class imbalances.

## Different Loss Functions to handle Class Imbalances in Long Tailed Data¶
### Paper Reference: 
"Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution" (EMNLP 2021) by Yi Huang, Buse Giledereli, Abdullatif Koksal, Arzucan Ozgur and Elif Ozkirimli. https://arxiv.org/abs/2109.04712

### Reference:
@inproceedings{huang2021balancing, title={Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution}, author={Huang, Yi and Giledereli, Buse and Koksal, Abdullatif and Ozgur, Arzucan and Ozkirimli, Elif}, booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)}, year={2021} }

#### Loss Functions for Multilabel Classification:
**Focal Loss**: Focal loss is used to address class imbalance by focusing more on hard-to-classify examples.
**Class Balanced Focal Loss**: Variation of focal loss adjusts for class imbalance by weighting the loss based on the frequency of each class.
**Distribution Balanced Loss**: Used to better handle the overlapping nature of multiple labels, by modeling the label distributions rather than treating labels as independent.
**Note**: More details can be found in the paper https://arxiv.org/abs/2109.04712
