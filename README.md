# forked-transformers
This repository contains fork of Transformer model from https://github.com/huggingface/transformers. Unlike original soruce code, which is framework, this repository is runnable stand-alone for language modeling task.

The source code on such open source library are great, however it is difficult to read all code for running model at a glance. For example, preprocessing data, define train-eval loop, integrating model in to the loop, these tasks are essential to write machine learning programs, but is it not always easy to looking source code from large open soruce project.

This repository combines open source model code and tutorial level preprocessing, and train-eval code. As a result, we expect readers can easily understand how Transformer models implemented and works from scratch.

## reference materials
This projects partly uses source code from following sources:
- Preprocessing, Train-Eval loop (except model): https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- Transformer model: [https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py](https://github.com/huggingface/transformers/blob/089cc1015ee73a2256b8bf3f89cd3abc3fb67d80/src/transformers/modeling_bert.py)
