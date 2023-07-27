---
layout: post
title: Transformer Explained in Depth with code 
date: 2023-07-04 08:57:00-0400
description: Re-implmentation of Paper Attention All you need Transformer
tags: formatting jupyter
categories: sample-posts
giscus_comments: true
related_posts: false
---

### Transfomers

* Now that you understand the main model components and the general idea, let's take a look at the full model.Intuitively, the model is exactly what we discussed before. In the encoder, the tokens communicate with each other and update their representation. In the decoder, the target token first looks at the previously generated target token, then the source, and finally updates its representation.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/blog.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}



<!-- {::nomarkdown}
{% assign jupyter_path = "assets/jupyter/note/Medical_imaging.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/note/Medical_imaging.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown} -->



Note that the jupyter notebook supports both light and dark themes.
