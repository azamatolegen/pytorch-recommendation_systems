# PyTorch Recommendation Systems

This repo contains a series of notebooks covering deep learning-based recommendation models.

The first part will cover different approaches of recommendation models using PyTorch on Jupyter/Google Colab notebooks.

The second part covers the serving of the best performed model using Fast API, Docker and Heroku.


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

The structure of the Part II is the following:
```
    data/               # dataset and dictionaries stored here
    docker/             # docker image
    flask/              # everything we need for flask api
        static/
        templates/
        api.py
        dockerfile
    models/             # stores our trained models
    runs/               # runs and logs
    download.py         # is the script that downloads and pre-processes the data
    loader.py           # is the script that loads the data into the model
    model.py            # is the model script that defines the Matrix Factorization model with biases
    train.py            # is the main training script. You can simply run python train.py to execute it
    reqiurements.txt
    README.md
```