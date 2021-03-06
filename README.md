# vae_models
Collection of python scripts running VAE models using Keras and Tensorflow as backend

This repository is a collection of VAE-scripts for image classification and generation and text classification and generation
(upcoming)

1. fashion_autoencoder.py has a python script that builds a VAE model using Keras for Fashion-MNIST dataset. Following are some
images that were generated by the network.

Representaion of images in test-set in two-dimensional latent space
![Representaion of original images in two-dimensional space](https://github.com/anktplwl91/vae_models/blob/master/latent_repr.png)

Some reconstructed images by Decoder network
![gen_images](https://github.com/anktplwl91/vae_models/blob/master/gen_sample.png)

A set of original input images and corresponding reconstructed images
![Input Images](https://github.com/anktplwl91/vae_models/blob/master/orig_imgs.png)
![Output Images](https://github.com/anktplwl91/vae_models/blob/master/gen_imgs.png)

There are three different convolutional layers in the Encoder network, each of them activates the input
image at different levels.

![orig_activations](https://github.com/anktplwl91/vae_models/blob/master/orig_activations.png)

Similar is the case for the Decoder network.
![gen_activations](https://github.com/anktplwl91/vae_models/blob/master/gen_activations.png)
