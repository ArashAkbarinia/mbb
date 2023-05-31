# Introduction to Deep learning
The teaching material for the deep learning course taught at the Mind, [Brain and Behaviour Master
programme](https://www.uni-giessen.de/de/studium/studienangebot/master/mbb?set_language=de) taught
at JLU Giessen.

Instructor: [Arash Akbarinia](https://arashakbarinia.github.io/)


## 1. [Setting up a deep learning environment](tutorials/environment_setup.md)

The prerequisite to continue with the rest of the materials is to set up a deep learning
environment. In this tutorial, we see how to do that using virtual environments.

## 2. [Basic operations in artificial neural networks](notebooks/basic_operations.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/basic_operations.ipynb)


Artificial neural networks consist of basic operations, such as convolution, pooling, and activation
functions. In this session, we cover those operations.

 * [Assignment 1](notebooks/assignment1.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/assignment1.ipynb)



## 3. [Building a deep neural network project](notebooks/build_DNN_project.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/build_DNN_project.ipynb)


In this session, we create a complete DNN project by constructing our own architecture and dataset.
We train our network for a simple classification problem.

 * [Assignment 2](notebooks/assignment2.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/assignment2.ipynb)



## 4. [Optimisation and learning](notebooks/optimisation_learning.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/optimisation_learning.ipynb)


In this session, we learn how a network acquires its knowledge and tunes its weights to perform a
certain task by exploring different loss functions in toy examples of 2D points.

 * [Assignment 3](notebooks/assignment3.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/assignment3.ipynb)


## 5. [Vision](notebooks/optimisation_learning.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/vision.ipynb)


In this session, we learn about the task of semantic segmentation (i.e., having a label for each 
pixel). We train a network to perform this task and we look into transfer-learning.

## 6. Deep generative models

In this session, we learn about deep generative models that can learn the distribution of data to generate new samples. We will explore three major generative models:
 * [Generative Adversarial Networks (GAN)](notebooks/gan.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/gan.ipynb)
 * [Variational Autoencoders (VAE)](notebooks/vae.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/vae.ipynb)
 * [Diffusion Probabilistic Models (DPM)](notebooks/dpm.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/dpm.ipynb)

## 7. Interpretation techniques

In this session, we learn about different interpretation techniques. How can we unravel the block 
box of deep neural networks? We will explore three techniques:
 * [Activation Map](notebooks/activation.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/activation.ipynb)
 * [Kernel Lesioning](notebooks/lesion.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/lesion.ipynb)
 * [Probing with Linear Classifiers](notebooks/linear_classifier_probe.ipynb) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArashAkbarinia/mbb/blob/main/notebooks/linear_classifier_probe.ipynb)