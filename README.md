# VAMPIRE

overall description

## Overview

Contributions

## Data Preparation

### Vessel Direction Traverse

- Setup

  [SAM-OCTA](https://github.com/ShellRedia/SAM-OCTA) is employed to generate initial vessel maps. Please set up the environment accordingly and download the pretrained weights to `vessel_traverse/sam_weights`.

- Segmentation

  ```shell
  cd vessel_traverse
  python test_sam_octa.py
  ```

- Patch Ordering

  ```shell
  python process_mask.py
  ```

### Vessel Morphology Description

We first employ a classification model trained on the [OCTA-500](https://ieee-dataport.org/open-access/octa-500) dataset to identify potential retinal diseases. 

Subsequently, we prompt GPT-4o with the diagnostic results to generate descriptions t on possible vascular morphologies.

Our generated description can be found in xx.

## Model Training

```shell
python finetune.py
```

