# Multi-source Transfer Learning for C–H Bond Activation Barrier Prediction

This repository contains the datasets and training codes used for the multi-source transfer learning prediction of C–H bond activation barriers.

The workflow includes multiple source-domain adsorption-energy prediction tasks and one target-domain regression task for C–H bond activation barriers.

## Repository Structure

```text
.
├── 270_DFT_3090_50_prediction-Ea.xlsx
├── 2079-CH3ads.xlsx
├── 6609-OHads.xlsx
├── 8635-Oads.xlsx
├── 8646-H_in_M_ads.xlsx
├── 13113-H_in_O_ads.xlsx
├── Source_1_200-6-layer-8635-Oads.py
├── Source_2_200-4-layer-13113-H_in_O_ads.py
├── Source_3_200-4-layer-8646-H_in_M_ads.py
├── Source_4_200-7-layer-6609-OHads.py
├── Source_5_1000-5-layer-2079-CH3ads.py
└── 3_Source_Target_Regression-Ea.py
```

## Dataset Description

### Target-domain dataset

| File | Description |
|---|---|
| `270_DFT_3090_50_prediction-Ea.xlsx` | Dataset for the target-domain regression task of C–H bond activation barrier prediction. It contains DFT-calculated activation barriers and structure-derived descriptors used for model training and prediction. |

### Source-domain datasets

| File | Source-domain property |
|---|---|
| `2079-CH3ads.xlsx` | Adsorption energy of CH3, denoted as `Eads(CH3)` |
| `6609-OHads.xlsx` | Adsorption energy of OH, denoted as `Eads(OH)` |
| `8635-Oads.xlsx` | Adsorption energy of O, denoted as `Eads(O)` |
| `8646-H_in_M_ads.xlsx` | Adsorption energy of H located on metal sites, denoted as `Eads(H@M)` |
| `13113-H_in_O_ads.xlsx` | Adsorption energy of H located on oxygen sites, denoted as `Eads(H@O)` |

These source-domain datasets were used to train individual feature extractors, which were then transferred to the target task of C–H bond activation barrier prediction.

In this work, the source-domain and target-domain datasets were split into training and test sets with a ratio of 9:1.

## Python code Description

| Python code | Description |
|---|---|
| `Source_1_200-6-layer-8635-Oads.py` | Training code for the source-domain feature extractor based on `Eads(O)` |
| `Source_2_200-4-layer-13113-H_in_O_ads.py` | Training code for the source-domain feature extractor based on `Eads(H@O)` |
| `Source_3_200-4-layer-8646-H_in_M_ads.py` | Training code for the source-domain feature extractor based on `Eads(H@M)` |
| `Source_4_200-7-layer-6609-OHads.py` | Training code for the source-domain feature extractor based on `Eads(OH)` |
| `Source_5_1000-5-layer-2079-CH3ads.py` | Training code for the source-domain feature extractor based on `Eads(CH3)` |
| `3_Source_Target_Regression-Ea.py` | Target-domain regression code for C–H bond activation barrier prediction using the fused multi-source representation |

