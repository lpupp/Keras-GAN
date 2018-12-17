
## Keras-GAN
Removed everything from Keras-GAN except discogan implementation.

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [DiscoGAN](#discogan)

## Installation
    $ git clone https://github.com/eriklindernoren/Keras-GAN
    $ cd Keras-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### DiscoGAN
Implementation of _Learning to Discover Cross-Domain Relations with Generative Adversarial Networks_.

[Code](discogan/discogan.py)

Paper: https://arxiv.org/abs/1703.05192

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan_architecture.png" width="640"\>
</p>

#### Example
```
$ cd discogan/
$ bash download_dataset.sh edges2shoes
$ python3 discogan.py
```   

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan.png" width="640"\>
</p>
