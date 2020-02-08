from flask import render_template
from flask import request

from app import app
from app import backend_api



#########################
## Front-end pages ######
#########################


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


#########################
## Back-end pages ######
#########################

@app.route('/backend/get_n_words')
def backend_get_n_words():
	args = request.args
	return backend_api.get_n_words(category=args['category'].upper().strip(),
								   n=args.get('n', 10))



