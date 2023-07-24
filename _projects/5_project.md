---
layout: page
title: Medical Application Signs Detection Image
description: End-To-End Web Application which implmented Training Models and Deploymenet using Docker and Streamlite
img: assets/img/project2/img.png
importance: 1
category: fun
---

This Project include all the Techno used Such Docker and TensorFlow

#### Project Intro
this repo contain the full the project DL with Web appplication using Streamlit for deployment 

#### Installation 

i provided the Docker Image to run Directly which contain all the app requimemnts ,
to run the Docker images following command :

```sh 
    docker pull ghcr.io/deep-matter/streamlit:latset 
```
```sh
    docker run -i -t streamlit:latset
```
### Model Traning 

here we used Convolution Neual network to develope a model for predicting the signs of patient based on the Eyes from Images the code can found in Colab [![Open Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o7USeyjTLjmgjjGXkQLt96HYNAbc_r7j)

### Docker-Compose 

also you can run app using docker-compose command follwing you need to keep docker-compose.yml in root folder 

```sh
    docker-compose up 
```

InterFace Application : 

<div align="center">
{% include figure.html path="assets/img/project2/logo.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>



The Full code in Github:  (read more about the <a href="https://github.com/deep-matter/Medical_Application_Streamlit/tree/main">Prohect Github</a> 