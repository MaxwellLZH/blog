from flask import render_template

from app import app
from app import backend_api


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/backend/<string:category>/get_words')
def backend_get_words(category):
	return backend_api.get_words(category=category.upper().strip())

