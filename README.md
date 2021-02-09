# PyTorch Recommendation Systems

This repo contains notebooks covering how to do recommendation using [PyTorch](https://github.com/pytorch/pytorch).

The first part will cover different approaches of recommendation models: . 

The second part covers the serving of the best performed model.

## Getting Started

Install tensorboard:

``` bash
pip uninstall tensorboard
```

Install wandb which we'll use for logging & dashboarding of our models:
``` bash
pip install wandb
```
In a terminal also login to wandb by runing wandb login. You'll have to setup an account if you haven't before.



## Part I

* 1 - [Matrix Factorization](https://github.com/azamatolegen/pytorch-recommendation_systems/blob/main/Part%20I/Matrix_Factorization_(MF).ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DnQjPm60UYM2HdhtqLfLHH_1IyprCfpV#scrollTo=FlcZ96-kuYyX)

    This notebook covers the workflow of a Matrix Factorization model in PyTorch. 
    Matrix Factorization is a class of the collaborative filtering algorithm. 
    The end goal of Matrix Factorization is basically to build a matrix of users and items filled with known and predicted ratings.
    
* 2 - [Factorization Machines](https://github.com/azamatolegen/pytorch-recommendation_systems/blob/main/Part%20I/Factorization_Machines_(FM).ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JP4qEEpkHg0UFwcWRujlN025J9Hkd-ya#scrollTo=M8IxcFFpGx1L)

    A downside of MF is that it is simply a matrix decomposition framework. 
    As such, we can only represent the matrix as a user-item matrix and unable 
    to incorporate side features such as movie genre, language, etc. 
    The factorization process has to learn all these from existing interactions. 
    Hence, factorization machines are introduced as an improved version of MF.

* 3 - [Deep Factorization Machines](https://github.com/azamatolegen/pytorch-recommendation_systems/blob/main/Part%20I/Deep_Factorization_Machines_(DeepFM).ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h2bl3xNda8yarVGJwUo_kcV9euF3EUYT#scrollTo=x-MKRhCyaxNU)

    For real-world data where inherent feature crossing structures are usually very complex and nonlinear, second-order feature interactions generally used in factorization machines in practice are often insufficient. Modeling higher degrees of feature combinations with factorization machines is possible theoretically but it is usually not adopted due to numerical instability and high computational complexity. One effective solution is using deep neural networks.    
## Part II 

* A - [Using TorchText with your Own Datasets](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb)

    The tutorials use TorchText's built in datasets. 

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- http://anie.me/On-Torchtext/
- http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
- https://github.com/spro/practical-pytorch
- https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
- https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
- https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
- https://github.com/Shawn1993/cnn-text-classification-pytorch
