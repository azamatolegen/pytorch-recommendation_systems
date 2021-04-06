# import al necessary libraries
import numpy as np
import pandas as pd
import joblib
import torch
from model import MF

# from fastapi import FastAPI, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# Define the Hyper-parameters
lr = 1e-2  # Learning Rate
k = 10  # Number of dimensions per user, item
c_bias = 1e-6  # New parameter for regularizing bias
c_vector = 1e-6  # Regularization constant
num_users = 6040
num_movies = 3706
model = MF(num_users, num_movies, k=k, c_bias=c_bias, c_vector=c_vector)
model.load_state_dict(torch.load('./models/mf_biases.pth'))
user_emb = model.user.weight.detach().numpy()
movie_emb = model.item.weight.detach().numpy()

user_mapping = joblib.load("./data/dict_data/user_mapping.pkl.zip")
item_mapping = joblib.load("./data/dict_data/item_mapping.pkl.zip")
movies_mapping = joblib.load("./data/dict_data/movies_mapping.pkl.zip")
inv_movies_mapping = joblib.load("./data/dict_data/inv_movies_mapping.pkl.zip")
inv_item_mapping = joblib.load("./data/dict_data/inv_item_mapping.pkl.zip")

# get the movie titles
movie_titles = list(movies_mapping.keys())

import json
from wtforms import TextField, Form
from flask import Flask, Response, render_template, request
app = Flask(__name__)

# SearchForm class will allow us to have autocomplete feature
class SearchForm(Form):
    movie_autocomplete = TextField('Movie name', id='movie_autocomplete')


@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
    '''
        autocomplete - function to respond to a request from javascript fuction
        responsible for autocomplete feature. Sends all the movie titles to the front.
    '''
    return Response(json.dumps(movie_titles), mimetype='application/json')


def to_recommend(requested_movie):
    if requested_movie in movies_mapping:
        idx = item_mapping[movies_mapping.get(requested_movie)]
        recommendation_list = sorted([(i, np.dot(movie_emb[idx], movie_emb[i])) for i in range(num_movies)], key=lambda x: x[1],reverse=True)[:10]
        recommendation_list = dict(recommendation_list)
        recommendation_list = sorted(recommendation_list, key=recommendation_list.get, reverse=True)

        movies_idx = []
        for idx in recommendation_list:
            movies_idx.append(inv_item_mapping[idx])

        similar_movies = []
        for idx in movies_idx:
            similar_movies.append(inv_movies_mapping[idx])

        return similar_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("index.html", form=form)

@app.route('/results', methods=['GET', 'POST'])
def results():
    requested_movie = request.args.get('movie_autocomplete', '')
    # get the recommendatuins on requested movie
    if requested_movie in movies_mapping:
        idx = item_mapping[movies_mapping.get(requested_movie)]
        recommendation_list = sorted([(i, np.dot(movie_emb[idx], movie_emb[i])) for i in range(num_movies)], key=lambda x: x[1],reverse=True)[:10]
        recommendation_list = dict(recommendation_list)
        recommendation_list = sorted(recommendation_list, key=recommendation_list.get, reverse=True)

        movies_idx = []
        for idx in recommendation_list:
            movies_idx.append(inv_item_mapping[idx])

        similar_movies = []
        for idx in movies_idx:
            similar_movies.append(inv_movies_mapping[idx])
    form = SearchForm(request.form)

    return render_template(
        'results.html',
        requested_movie=requested_movie,
        similar_movies=similar_movies,
        form=form
    )

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    main()