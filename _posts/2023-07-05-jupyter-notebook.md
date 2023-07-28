---
layout: post
title: Generating New Drug SMILES Data Using Model RNN-LTSM
date: 2023-07-04 08:57:00-0400
description: <b> De novo simply means to synthesize new </b> . The idea is the train the model to learn patterns in SMILES strings so that the output generated can match valid molecules

tags: formatting jupyter
categories: sample-posts
giscus_comments: true
related_posts: false
---

### De novo simply means to synthesize LTSM-RNN

* My current task involves training a Recurrent Neural Network (RNN) to generate new molecules, known as de novo synthesis. The RNN learns patterns from SMILES strings, which are string representations of molecules based on their structure and components. This approach enables the model to produce valid molecules as output, as SMILES provides a computer-friendly representation of molecules.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/blog_chemi.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog_chemi.ipynb %}{% endcapture %}
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
