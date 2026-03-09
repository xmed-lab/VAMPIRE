# VAMPIRE: Uncovering Vessel Directional and Morphological Information from OCTA Images for Cardiovascular Disease Risk Factor Prediction

<a href="https://papers.miccai.org/miccai-2025/paper/3405_paper.pdf"><img src="https://img.shields.io/badge/Paper-MICCAI2025-green.svg?style=flat-square"></a>
<a href="https://huggingface.co/datasets/Luxuriant16/OCTA-CVD"><img src="https://img.shields.io/badge/Dataset-OCTACVD-orange.svg?style=flat-square"></a>


VAMPIRE, Vessel-Aware Mamba-based Prediction model with Informative Enhancement, is a novel multi-purpose paradigm of CVD risk assessment that jointly performs CVD risk and CVD-related condition prediction, aligning with clinical experiences.

## Updates
- OCTA-CVD dataset is released [here](https://huggingface.co/datasets/Luxuriant16/OCTA-CVD).

## Overview
VAMPIRE extracts crucial vascular characteristics through two key components: 
- a Mamba-Based Directional (MBD) Module that captures fine-grained vascular trajectory features 
- an Information-Enhanced Morphological (IEM) Module that incorporates comprehensive vessel morphology knowledge.

<div align="center">
<img src="asset/framework.png" width="65%" />
</div>

## Data Preparation
### Download OCTA-CVD
OCTA-CVD can be downloaded at [link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lwangdk_connect_ust_hk/EZK2Ez1_A_tDmU-WevHcYlEBlXwXdTGAdL6wvE-ASaVvNQ?e=NsR9Sn). After downloading the data, move them into `data` directory. The structure should be like the following:
```
.
└── data/
    ├── OCTA-Enface/
    │   ├── Choriocapillaris
    │   ├── Deep
    │   └── Superficial
    └── Label/
        └── folds_info
```

### Vessel Direction Traverse

- Setup

  [SAM-OCTA](https://github.com/ShellRedia/SAM-OCTA) is employed to generate initial vessel maps. Please set up the environment accordingly and download the pretrained weights to `vessel_traverse/sam_weights`.

- Segmentation

  ```shell
  cd vessel_traverse
  python test_sam_octa.py
  ```

  Then, the vessel segmentation map would be saved into `seg` directory.

- Patch Ordering

  ```shell
  python process_mask.py
  ```

  Then, we can obtain `img2order.pkl`, which records the traverse order for each OCTA scan.

### Vessel Morphology Description

We first employ a classification model trained on the [OCTA-500](https://ieee-dataport.org/open-access/octa-500) dataset to identify potential retinal diseases. 

Subsequently, we prompt GPT-4o with the diagnostic results to generate descriptions on possible vascular morphologies. The prompt can be referred to `vessel_descrp/disease_prompts.json`.

Our generated description can be found in `vessel_descrp/p2res_disease.json`.

## Environment Setup

- Create Environment

  ``` shell
  conda create -n vampire python=3.9.21 -y
  conda activate vampire
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  pip install numpy==1.21.6
  pip install scikit-learn==1.2.2
  pip install transformers==4.47.1
  ```

- Install Mamba
  
  Mamba requirements `causal_conv1d` and `mamba-1p1p1` are built from [Vim](https://github.com/hustvl/Vim)

- Prepare pretrained weights

  We use pretrained fundus weights from [VisionFM](https://github.com/ABILab-CUHK/VisionFM). Please first download the [weights](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view) and save into the `pretrain` directory.

## Model Training
Run training with `finetune.py`, and the evaluation will be executed sequentially after training.

```shell
python finetune.py
```
## Citation
```
@InProceedings{
  VAMPIRE_MICCAI2025,
  author = { Wang, Lehan AND Wang, Hualiang AND Ou, Chubin AND Chen, Lushi AND Liang, Yunyi AND Li, Xiaomeng },
  title = { { VAMPIRE: Uncovering Vessel Directional and Morphological Information from OCTA Images for Cardiovascular Disease Risk Factor Prediction } }, 
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year = {2025},
  publisher = {Springer Nature Switzerland},
  volume = { LNCS 15974 },
  month = {October},
  pages = { 649 -- 659 },
}
```



