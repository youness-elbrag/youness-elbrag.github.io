---
layout: page
title: Transformer From Scratch 
description:  Implmentation of -Transformer following Series of Article About Vision-Langauge model , in project i implmented from Scratch using numpy
img: assets/img/project10/full.png
importance: 6
category: work
---


### Transformers Numpy 

this is Implementation of Transfomrers Numpy Version from Scratch which all LLM based on have abetter understaning cam help to build this type of model in right concept 
Welcome to the repository for the TransformersNumpy-Version project! This project focuses on implementing Transformers, a groundbreaking model in the field of natural language processing and machine learning, using the powerful numpy library.

<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/full.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### Introduction

Transformers have revolutionized the way we process and understand natural language, enabling breakthroughs in tasks such as machine translation, sentiment analysis, and question-answering systems. This repository aims to provide a comprehensive implementation of Transformers using numpy, showcasing the core concepts and functionalities of this powerful model.

###  Key Features

- **Numpy Implementation:** The implementation in this repository heavily relies on the numpy library, allowing for efficient computations and easy-to-understand code.
- **Full Implementation:** The repository provides a complete implementation of Transformers, including attention mechanisms, positional encoding, and feed-forward networks.
.

###  Contents

Here's an overview of the contents you'll find in this repository:

- **`Decoder.py and Encoder.py`:** This file contains the main implementation of the Transformer model using numpy. It includes classes and functions for attention mechanisms, positional encoding, and the overall Transformer architecture.
- **`LayersNumpy.py`:** This file provides utility functions for data preprocessing and handling, enabling seamless integration with different datasets.
- **`transfomers-explained-mathematic-and-code-in-depth.ipynb`:** This Jupyter Notebook serves as an example to showcase how to use the Transformer model implemented in this repository. It includes a step-by-step walkthrough of training and evaluating the model on a specific task.
- **`main.py`:** this provide full combinations of the blocks Transformser.

### Self-Attention Math

<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/att.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

```python

def Self_Attention(input_embedding ,WieghtMatrix_QKY, out_wieghts,mask=None,batch_first=True) :
    """
    Self-Attention take input of emebeding matrix which asseccoite with 
    the Positional encoding ww will cover later in section 
    Query and Key and Value all of them have the same dimession as the input 
    """
    try : 
        if batch_first==True:
            Query , Key , value = np.split(input_embedding@WieghtMatrix_QKY , 3 , axis=-1)
            if mask is not None:
                assert mask.shape[0] == input_embedding.shape[1],\
                    f"input dimession of mask doesn't match with dimession of embedding input:{mask.shape[0]} {input_embedding.shape[0]}"
                Attention = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(input_embedding.shape[-1]) + mask) 
                return  Attention@value@out_wieghts , Attention
            else:
                Attention = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(input_embedding.shape[-1])) 
                return  Attention@value@out_wieghts , Attention
    except:
        raise Exception("Batch argumment is missing")

```


### 1.2 Multi-Head Attention


<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/heads.gif" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

 
```python
def multiHeads_Attention(input_embedding ,wieghtsMatrix_QKY , heads ,out_Wieght , mask=None):
    B , seq_len , embed_size = input_embedding.shape
    # we have dim input of B . seq_len , 
    # embed_size ==> B, seq_len , embed_size/ heads 
    #=> Swape axis into [batch , heads , seq_len , embe_size / heads ]
    Query , Key, Value = np.split(input_embedding@wieghtsMatrix_QKY,3, axis=-1)
    Query , Key, Value = [a.reshape(B , seq_len ,heads , (embed_size // heads)).swapaxes(1,2) for a in (Query , Key, Value)]
    if mask is not None:
        atten = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(embed_size // heads) + mask ) 
        return (atten@Value).swapaxes(1,2).reshape(B , seq_len , embed_size)@out_Wieght , atten
    else:
        atten = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(embed_size // heads)) 
        return (atten@Value).swapaxes(1,2).reshape(B , seq_len , embed_size)@out_Wieght , atten
```
 
### 1.3 Scale Dot Product Attention


<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/mha_img_original.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

 
```python
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self,config):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(config["attention_droput"])

    def forward(self, q, k, v,output_attentions=False):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        score =  self.attention_dropout(score)

        # 4. multiply with Value
        v = score @ v
        if not output_attentions:
            return (v, None)

        return (v, score)
```
### Positional Encodeing 


<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/pos.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

```python 

import torch.nn as nn 
import numpy as np 

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding,self).__init__()    
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]  

```

###  Residual Cennection

<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/res.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

```python

class ResidualCennection(nn.Module):
    def __init__(self,x, residual):
        super(ResidualCennection,self).__init__()
        self.pass_trough = x
        self.addtion = residual
    def forward(self,*args, **kwargs):
        x = self.pass_trough
        return x + self.addtion
```
 
### 1.4 Layer Norm And Softmax


<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/layer_norm.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

 
```python


def NormLayar(Z):
    mean = Z.mean(axis=-1 , keepdims=True)
    var = Z.var(axis=-1 , keepdims = True)
    return ((Z - mean) / np.sqrt(var)) + eps

def ReLU(Z):
    return np.maximum(Z,0)
    

def Softmax(z):
    """
    descriptions function : Softmax is non-linear function that give the averege of between 
    0 and 1 of in element in matrix 
    """
    e_x = np.exp(z - z.max(axis=-1,keepdims=True))
    return e_x / np.sum(e_x , axis=-1 ,keepdims=True)

```

### 1.5 MLP

    
```python

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["embedding_size"],config["mlp_ratio"] * config["embedding_size"])
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(config["embedding_size"] * config["mlp_ratio"], config["embedding_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

```

### 1.6 TransformerEncoder

<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/enc.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
    
```python
import numpy as np 
from layer import * 
from layerNumpy import *

def TransfomerEncoder(embed_input,mask ,
                      head, Wieghts_QKY , 
                      Wieghts_out ,FullyLinear1, 
                      FullLinear2 , eps):
    input_embedding = embed_input.numpy()
    multiHeads , _ = multiHeads_Attention(input_embedding,
                                     Wieghts_QKY.detach().numpy().T,
                                     head,
                                     Wieghts_out.detach().numpy().T,  
                                     mask=None)
    Residual = NormLayar((input_embedding + multiHeads) + eps )
    
    output = NormLayar((Residual + ReLU(np.matmul(Residual,FullyLinear1))@FullLinear2) + eps)
    return output
```
### 1.6 TransformerDecoder

<div class="row" >
    <div class="col-sm mt-3 mt-md-0" align="center" >
    {% include figure.html path="assets/img/project10/dec.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

```python
import numpy as np 
from layer import *
from layerNumpy import *


def TransfomerDecoder(embed_input,mask ,
                      head, Wieghts_QKY , 
                      Wieghts_out ,FullyLinear1, 
                      FullLinear2 , eps,enc=None):
    input_embedding = embed_input.numpy()
    multiHeads , _ = multiHeads_Attention(input_embedding,
                                     Wieghts_QKY.detach().numpy().T,
                                     head,
                                     Wieghts_out.detach().numpy().T,  
                                     mask=None)
    Residual = NormLayar((input_embedding + multiHeads) + eps )
    
    if enc is not None:
        Query , key  = np.split(enc , 2 , axis=-1)
        enc_ = np.concatenate((Query[:,:,:16] , key[:,:,:16]  ,Residual[:,:,:32] ), axis = -1)
        MaskedMUltiHeads ,_= multiHeads_Attention(enc_,
                                         Wieghts_QKY.detach().numpy().T,
                                         head,
                                         Wieghts_out.detach().numpy().T,  
                                         mask=mask.numpy())
        Residual = NormLayar((enc + MaskedMUltiHeads) + eps )
    
    output = NormLayar((Residual + ReLU(np.matmul(Residual,FullyLinear1))@FullLinear2) + eps)
    return output

```






### Transfomer model 
 
```python 

from blocks.encoder import * 
from blocks.decoder import * 
import numpy as np

class Transformer:
    def __init__(self, embedding_size , heads):
        super(Transformer,self).__init__()
        
        """
        Blocks models
        -------------
        encoder : is used without the Mask at this stage 
        Decoder : include the Mask and with trick to split the Qurey and Key 
        
        Parametes :
        Wieghst_QKY_Encoder : is Leanrble wieght matrix feed into Encoder
        Wieghst_QKY_Decoder : is Leanrble wieght matrix feed into ncoder
        embed_size : is mebeding_size dimession of input 
        Seq_len : max lenght of vacolublaries 
        Linear_1 : Linear Denes layar of Deooder
        Linear_2 : Linear Denes layar of Deooder after include the ooutput from Encoder
        Linear_encoder : Linear Denes layar of Enooder
        """
    
        self.embed_input = embedding_size
        self.heads = heads 
        self.eps = 1e-12
        self.W_QKY_Encoder = transEncoder.self_attn.in_proj_weight
        self.W_Out_Encoder = transEncoder.self_attn.out_proj.weight
        self.W_QKY_Decoder = transDecoder.self_attn.in_proj_weight
        self.W_Out_Decoder = transDecoder.self_attn.out_proj.weight
        self.Linear1_Encoder = transEncoder.linear1.weight
        self.Linear2_Encoder = transEncoder.linear2.weight
        self.Linear1_Decoder = transDecoder.linear1.weight
        self.Linear2_Decoder = transDecoder.linear2.weight
    
    def forward(self, enc_ , dec_, mask):
        Encoder_= TransfomerEncoder(enc_ ,None ,
                  self.heads, self.W_QKY_Encoder , 
                  self.W_Out_Encoder ,self.Linear1_Encoder.detach().numpy().T, 
                  self.Linear2_Encoder.detach().numpy().T , self.eps)

        Decoder_ = TransfomerDecoder(dec_, mask ,
                  self.heads, self.W_QKY_Decoder , 
                  self.W_Out_Decoder , self.Linear1_Decoder.detach().numpy().T, 
                  self.Linear2_Decoder.detach().numpy().T ,self.eps,enc=Encoder_)

        return Decoder_


```