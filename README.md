# Text Matching Improves Sequential Recommendation by Reducing Popularity Biases

Source code for CIKM 2023 Text Matching Improves Sequential Recommendation by Reducing Popularity Biases

Click the links below to view our papers, checkpoints and datasets

<a href='https://arxiv.org/abs/2308.14029'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a><a href='https://huggingface.co/OpenMatch/TASTE-beauty'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-beauty-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-sports'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sports-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-toys'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-toys-blue'></a><a href='https://huggingface.co/OpenMatch/TASTE-yelp'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-yelp-blue'></a><a href='https://drive.google.com/drive/folders/1U_tkCJq80kdefV9z_FdWDCMWvtUrm0or?usp=sharing'><img src='https://img.shields.io/badge/Google Drive-Dataset-yellow'></a> 


## Getting Started
### Installation

**1. Install the following packages using Pip or Conda under this environment**

```
python>=3.8
transformers==4.22.2
numpy==1.23.5
datasets==2.11.0
faiss-cpu
scikit-learn>=1.1.2
pandas
tensorboard
```

We provide the version file `reproduce/environment.yml` of all our used packages, if you have any problems configuring the environment, please refer to this document.

**2. Install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.**


```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install .
```

We do not include all the requirements in the package. You may need to manually install `torch`


**3. Prepare the pretrained T5 weights**

TASTE is built on [T5-base](https://huggingface.co/t5-base/tree/main).



### Reproducibility

We provide all scripts, log files, and tensorboard required for dataset processing, training, evaluation, and testing under the reproduce folder. Please refer to these files and the readme in the `reproduce folder` to reproduce.



## Acknowledgement

+ [OpenMatch](https://github.com/OpenMatch/OpenMatch) This repository is built upon OpenMatch! An all-in-one toolkit for information retrieval!
+ [FID](https://github.com/facebookresearch/FiD) The model architecture of TASTE refers to FID!
+ [RecBole](https://github.com/RUCAIBox/RecBole) Data processing and evaluation code based on RecBole! RecBole is developed based on Python and PyTorch for reproducing and developing recommendation algorithms in a unified, comprehensive and efficient framework for research purpose!


If you're using TASTE in your research or applications, please cite using this BibTeX:
```bibtex

```


## Contact

If you have questions, suggestions, and bug reports, please email:
```
meisen@stumail.neu.edu.cn
```
