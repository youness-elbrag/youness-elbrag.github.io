---
layout: page
title: Pyramid Position Encoding Generator
description: this porject is supported by Jordan University of science and Technology , Multi-instance Learning approach based on Transformer
img: assets/img/project6/img.png
importance: 4
category: work
---

# Coronaries Arteries Diseases Weakly supervised Learning based method 
this porject is supported by Jordan University of science and Technology , alonge side with this research we explored a paper that introduced [Multi-instance Learning approach based on Transformer](https://arxiv.org/abs/2106.00908) in our mission we Re-design PPGE method for more efficient training by improvig **Convolution with Fast Fourier Transform** which called the method Fast Fourier Postional encoding **FFPE**

**Notation:** the implementation still under progres as long as we are trying to collect dataset of Coronaries-Arteries-Diseases
now tried to test the approach on Data from [Kaggle RSNA Screening Mammography Breast Cancer Detection](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge)
### Introduction 

Multiple instance learning (MIL) is a powerful tool to solve the weakly supervised classification in whole slide image (WSI) based pathology diagnosis. However, the current MIL methods are usually based on independent and identical distribution hypothesis, thus neglect the correlation among different instances. To address this problem, we proposed a new framework, called correlated MIL, and provided a proof for convergence. Based on this framework, we devised a Transformer based MIL (TransMIL) used within Fast Fourire to Enhanced Pyramid Position Encoding Generator Projection Linear System , 

<div align="center">
{% include figure.html path="assets/img/project6/img.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

* Setup the ENV:
     - Create the environment 

            conda create --name TransFFT-MIL python=3.6
    - install the requirements
    
            pip install -r requirements.txt  

* Run the code :
    -  training model </br>
        **Note** in our experiment we Re-Developed two approaches based Positional Encodings methods **FFTPEG** and **FF_ATPEG**
        that can be changed in TransFFPEG.py file 

```python
     python train.py --stage 'train' --gpus 0 --Epochs 200
```       


The Full code in Github:  (read more about the <a href="https://github.com/deep-matter/TransFFT-MILCoronaries-Arteries/tree/main">Prohect Github</a> 