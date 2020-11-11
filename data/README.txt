This folder consists of the local and global quality estimations by ResNetQA models for CASP12 and CASP13 decoys.

ResNetQA:        The model trained using local and global MSE loss in terms of S-score and GDT_TS.
ResNetQA-R:      The model trained using local and global MSE loss plus global margin ranking loss in terms of S-score and GDT_TS.
ResNetQA-lDDT:   The model trained using local and global MSE loss in terms of local and global lDDT.
ResNetQA-R-lDDT: The model trained using local and global MSE loss plus global margin ranking loss in terms of local and global lDDT.

CASP12_stage2/
    ResNetQA/
    ResNetQA-R/
    ResNetQA-lDDT/
    esNetQA-R-lDDT/

CASP13_stage2/
    ResNetQA/
    ResNetQA-R/
    ResNetQA-lDDT/
    esNetQA-R-lDDT/

Evaluation.py: The script to evaluate the performances of different models.