# Multi-source Transfer Learning for C–H Bond Activation Barrier Prediction
This repository contains the datasets, source-domain training codes, target-domain regression codes, and corresponding Jupyter notebooks used for multi-source transfer learning prediction of methane C–H bond activation barriers.
The workflow consists of multiple source-domain adsorption-energy prediction tasks and one target-domain regression task for C–H bond activation barrier prediction. Source-domain models are first trained on adsorption-energy datasets, and the learned representations are then transferred and fused for the target-domain activation barrier prediction.
---
## Repository Structure
```text
.
├── 13113-H_in_O_ads.xlsx
├── 2079-CH3ads.xlsx
├── 270_DFT_3090_50_prediction-Ea.xlsx
├── 6609-OHads.xlsx
├── 8635-Oads.xlsx
├── 8646-H_in_M_ads.xlsx
│
├── Source_1_200-6-layer-8635-Oads.py
├── Source_1_200-6-layer-8635-Oads.ipynb
├── Source_2_200-4-layer-13113-H_in_O_ads.py
├── Source_2_200-4-layer-13113-H_in_O_ads.ipynb
├── Source_3_200-4-layer-8646-H_in_M_ads.py
├── Source_3_200-4-layer-8646-H_in_M_ads.ipynb
├── Source_4_200-7-layer-6609-OHads.py
├── Source_4_200-7-layer-6609-OHads.ipynb
├── Source_5_1000-5-layer-2079-CH3ads.py
├── Source_5_1000-5-layer-2079-CH3ads.ipynb
│
├── 3_Source_Target_Regression-Ea.py
├── 3_Source_Target_Regression-Ea.ipynb
│
├── Source_1_8635_Oads_270_3439_Ea_10_200_6layer_...
├── Source_2_13113_H_in_O_ads_270_3439_Ea_10_...
├── Source_3_8646_H_in_M_ads_270_3439_Ea_10_...
├── Source_4_6609_OHads_270_3439_Ea_10_200_7layer_...
├── Source_5_2079_CH3ads_270_3439_Ea_10_200_5layer_...
├── 3_Source_3439_270_51_Ea-merged-2000-20-0.1
│
└── README.md
The .py files provide executable Python scripts, while the corresponding .ipynb files provide Jupyter Notebook versions for interactive training, testing, and visualization.

The folders beginning with Source_1, Source_2, Source_3, Source_4, Source_5, and 3_Source contain the corresponding training outputs, transferred features, saved models, or regression results generated during source-domain pretraining and target-domain prediction.

Dataset Description
Target-domain dataset
File	Description
270_DFT_3090_50_prediction-Ea.xlsx	Target-domain dataset for methane C–H bond activation barrier prediction. It contains DFT-calculated activation barriers and structure-derived descriptors used for target-domain regression and prediction.
The target property is the C–H bond activation barrier.

Source-domain datasets
File	Source-domain property	Notation
8635-Oads.xlsx	Adsorption energy of O	Eads(O)
13113-H_in_O_ads.xlsx	Adsorption energy of H located on oxygen sites	Eads(H@O)
8646-H_in_M_ads.xlsx	Adsorption energy of H located on metal sites	Eads(H@M)
6609-OHads.xlsx	Adsorption energy of OH	Eads(OH)
2079-CH3ads.xlsx	Adsorption energy of CH3	Eads(CH3)
These source-domain datasets were used to train individual adsorption-energy prediction models. The feature extractors learned from these source tasks were transferred to the target task for C–H bond activation barrier prediction.

Python Code Description
Source-domain pretraining codes
File	Description
Source_1_200-6-layer-8635-Oads.py	Source-domain training code for the adsorption energy of O, Eads(O).
Source_1_200-6-layer-8635-Oads.ipynb	Jupyter Notebook version of the Eads(O) source-domain training code.
Source_2_200-4-layer-13113-H_in_O_ads.py	Source-domain training code for the adsorption energy of H located on oxygen sites, Eads(H@O).
Source_2_200-4-layer-13113-H_in_O_ads.ipynb	Jupyter Notebook version of the Eads(H@O) source-domain training code.
Source_3_200-4-layer-8646-H_in_M_ads.py	Source-domain training code for the adsorption energy of H located on metal sites, Eads(H@M).
Source_3_200-4-layer-8646-H_in_M_ads.ipynb	Jupyter Notebook version of the Eads(H@M) source-domain training code.
Source_4_200-7-layer-6609-OHads.py	Source-domain training code for the adsorption energy of OH, Eads(OH).
Source_4_200-7-layer-6609-OHads.ipynb	Jupyter Notebook version of the Eads(OH) source-domain training code.
Source_5_1000-5-layer-2079-CH3ads.py	Source-domain training code for the adsorption energy of CH3, Eads(CH3).
Source_5_1000-5-layer-2079-CH3ads.ipynb	Jupyter Notebook version of the Eads(CH3) source-domain training code.
Target-domain regression code
File	Description
3_Source_Target_Regression-Ea.py	Target-domain regression code for methane C–H bond activation barrier prediction using fused multi-source representations.
3_Source_Target_Regression-Ea.ipynb	Jupyter Notebook version of the target-domain regression code.
The target-domain model integrates transferred representations from multiple source-domain adsorption-energy models and performs regression for C–H bond activation barrier prediction.

Workflow
The overall workflow is:

Prepare source-domain datasets

8635-Oads.xlsx
13113-H_in_O_ads.xlsx
8646-H_in_M_ads.xlsx
6609-OHads.xlsx
2079-CH3ads.xlsx
Train source-domain feature extractors

Train individual neural networks for each adsorption-energy prediction task.
Save the learned source-domain models or representations.
Prepare target-domain dataset

270_DFT_3090_50_prediction-Ea.xlsx
Transfer and fuse source-domain representations

Extract useful representations from pretrained source-domain models.
Fuse multi-source information for the target-domain regression task.
Train target-domain regression model

Use 3_Source_Target_Regression-Ea.py or 3_Source_Target_Regression-Ea.ipynb.
Predict methane C–H bond activation barriers.

Requirements
The codes are written in Python and Jupyter Notebook. The main packages include:
Python
NumPy
Pandas
Scikit-learn
PyTorch
Matplotlib
OpenPyXL
Jupyter Notebook
A typical environment can be installed using:
pip install numpy pandas scikit-learn torch matplotlib openpyxl notebook
Depending on the local CUDA environment, please install the appropriate version of PyTorch from the official website:
https://pytorch.org/
Usage
1. Train source-domain models
For example, to train the source-domain model for Eads(O):
python Source_1_200-6-layer-8635-Oads.py
To train the other source-domain models:
python Source_2_200-4-layer-13113-H_in_O_ads.py
python Source_3_200-4-layer-8646-H_in_M_ads.py
python Source_4_200-7-layer-6609-OHads.py
python Source_5_1000-5-layer-2079-CH3ads.py
Alternatively, the corresponding .ipynb files can be opened and executed in Jupyter Notebook.

2. Train the target-domain regression model
After source-domain pretraining, run:
python 3_Source_Target_Regression-Ea.py
or execute:
3_Source_Target_Regression-Ea.ipynb
in Jupyter Notebook.

Citation
If you use this repository, datasets, models, or codes in your research, please cite the following work:
Wangqiang Lin, Huiyang Zhang, Jinxin Sun, Chongyi Ling, Xiuyun Zhang, Qiang Li, Qionghua Zhou, and Jinlan Wang.
Mechanism-Guided Catalyst Discovery for Methane C-H Activation via Structure-Aware Multi-Source Transfer Learning. 2026.
Manuscript under revision.
The citation information will be updated after the paper is formally accepted and published.

Notes
The datasets are provided in .xlsx format.
The .py files are script versions for direct execution.
The .ipynb files are notebook versions for interactive inspection and reproduction.
The output folders store source-domain and target-domain training results, including intermediate files, saved models, or prediction outputs generated by the corresponding scripts.
