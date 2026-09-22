# EZ-Ar-VC: Robust Arabic Voice Anonymization with Generalization across Dialects and Code-Switching

[![Python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/EZ-AR-VC/EZ-AR-VC)
[![Demo](https://img.shields.io/badge/GitHub-Demo%20Page-orange.svg)](https://EZ-AR-VC.github.io/EZ-AR-VC-DEMO/)


## Installation

Create the conda environment:

```bash
conda create -n ez-ar-vc python=3.10
conda activate ez-ar-vc
```

</details>

## Local Installation

Clone the repository and initialize the submodules:

```bash
git clone https://github.com/EZ-AR-VC/EZ-AR-VC
cd EZ-AR-VC

git submodule update --init --recursive

pip install -e .
```

Install ESPnet for XEUS using the required version:

```bash
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
```

## Download Checkpoints and K-Means Models

Download the pretrained checkpoints and K-Means models from:

[**Download Checkpoints & K-Means Models**](https://drive.google.com/drive/folders/1ejJKNULoPWbU84fCdkXsxw9K1sPHKAct?usp=sharing)

Place the downloaded files inside the `checkpoints/` directory:

```text
EZ-AR-VC/
├── checkpoints/
│   └── ...
├── src/
├── README.md
└── ...
```

## Inference

We provide a Jupyter notebook for running inference.

Open:
 
[src/f5_tts/infer/inference.ipynb](src/f5_tts/infer/inference.ipynb).

Run the notebook cells sequentially. The converted audio will be available in the final cell.
  

## Acknowledgements

This project builds upon the following open-source projects:

- [F5-TTS](https://arxiv.org/abs/2410.06885)
- [EZ-VC](https://arxiv.org/pdf/2505.16691)
 

We thank the authors and contributors of these projects for making their research and implementations publicly available.
 
