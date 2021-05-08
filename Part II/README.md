A web app that recommends movies similar to those a user provides as an input.

## Data
[MovieLens1M Dataset](https://grouplens.org/datasets/movielens/1m/): a famous dataset within the recommendation systems research community. The data contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

## Containerize : Docker
If you want to check out the container yourself, here is a link to my [Docker Hub](https://hub.docker.com/r/azamatolegen/pytorch-recommendation_systems-part2). You can pull the image by:
```
docker pull azamatolegen/pytorch-recommendation_systems-part2
```
## Deploy : Heroku
The app is hosted at https://recsys-part2.herokuapp.com/

## The project stricture
```
    app/                # api
        models/         # stores our trained models
        static/         # styles and
        templates/      # templates for web page
        api.py          # flask api to interact with the model
    data/               # dataset and dictionaries stored here
        dict_data/      # data stored in dictionary format
        download.py     # download and pre-process
    training/           # model training scripts
        runs/           # runs and logs
        loader.py       # the script that loads the data into the model
        model.py        # the model script that defines the Matrix 
        train.py        # the main training script
    dockerfile          # docker image    
    reqiurements.txt    # requirements listed in this file
    README.md
```