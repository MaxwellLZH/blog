import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go

from typing import List
import os
import subprocess


if not os.path.exists('./kaggle.db'):
	print('Downloading notebook information from Kaggle ...')
	subprocess.call('python ./scrapper.py') 
	print('Download complete :)')

	
engine = create_engine('sqlite:///kaggle.db')


df_competition = pd.read_sql_table('competition', engine).rename(columns={'title': 'competitionName'})
df_kernel = pd.read_sql_table('kernel', engine).sort_values('totalVotes', ascending=False)
df_kernel = df_kernel.merge(df_competition[['id', 'competitionName', 'categories']], 
						left_on=['competitionId'],
						right_on='id')


print(df_competition.dtypes)


########################################## 
# define a bunch of filtering component  #
########################################## 
language_options = df_kernel.languageName.unique()

language_selector = dcc.Dropdown(
						id='language',
					    options=[
					    	{'label': l, 'value': l} for l in language_options
					    ],
					    value='Python',
					    multi=True,
					) 


category_options = ','.join(df_competition['categories'])
category_options = list(set([c for c in category_options.split(',') if c.strip() != '']))

category_selector = dcc.Dropdown(
						id='category',
					    options=[
					    	{'label': l, 'value': l} for l in category_options
					    ],
					    value=['tabular data', 'text data'],
					    multi=True,
					)


competition_options = list(df_competition.competitionName.unique())
competition_selector = dcc.Dropdown(
						id='competition',
					    options=[
					    	{'label': l, 'value': l} for l in competition_options
					    ],
					    value=df_competition.sort_values('totalCompetitors', ascending=False)['competitionName'].head(3).tolist(),
					    multi=True,
					)



#####################
##  Page Layout.   ##
#####################


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Learn from top Kaggle notebooks!'),

    # html.Div([
    # 	language_selector,
    # 	category_selector
    # 	]),
   

    html.H3(children='List of competitions'),

	html.Div(children=['Choose dataset category:', category_selector]),

    dcc.Graph(id='fig-competition',

    	style={'border-spacing': '10px'}
    	),


    html.H3(children='List of kaggle notebooks'),

    html.Div(children=['Choose competition', competition_selector], style={'margin-bottom': '10px'}),
	html.Div(children=['Choose language:', language_selector], style={'magin-bottom': '10px'}),


    dcc.Graph(id='fig-kernel')
    
], style={'margin': '40px'})


##########################################
## All the backend filtering operations ##
##########################################

competition_cols = ['competitionName', 'briefDescription', 'totalCompetitors', 'categories']
kernel_cols = ['competitionName', 'title', 'totalVotes', 'languageName', 'notebookFullUrl']


def _contains(s: pd.Series, keywords: List[str]):
	""" Check if any of the keywords is contained in s 
		s is seperated by comma 
	"""
	s = s.apply(lambda x: set(x.split(',')))
	keywords = set(keywords)

	return s.apply(lambda x: len(x & keywords) > 0)


def debug(f):
	""" Dummy debugger that prints function name and arguments """

	def wrapped(*args, **kwargs):
		print(f.__name__, args, kwargs)
		return f(*args, **kwargs)

	return wrapped()


def as_list(l):
	return [l] if not isinstance(l, list) else l


@app.callback(
	Output('fig-competition', 'figure'),
	[
	 Input('category', 'value')
	])
def update_competition_figure(category: List[str]):
	category = as_list(category)

	d = df_competition[_contains(df_competition.categories, category)]
	d = d.sort_values('totalCompetitors', ascending=False)

	fig_competition = go.Figure(data=[go.Table(
	    header=dict(values=competition_cols,
	                fill_color='paleturquoise',
	                align='left'),
	    cells=dict(values=[d[c] for c in competition_cols],
	               fill_color='lavender',
	               align='left'))
	])

	return fig_competition



@app.callback(
	Output('fig-kernel', 'figure'),
	[
	Input('language', 'value'),
	Input('category', 'value'),
	Input('competition', 'value')
	]
	)
def update_kernel_figure(language: List[str], category: List[str], competition: List[str]):
	language, category, competition = as_list(language), as_list(category), as_list(competition)

	d = df_kernel[(_contains(df_kernel.categories, category)) \
					& (df_kernel.languageName.isin(language)) \
					& (df_kernel.competitionName.isin(competition))]
	d = d.sort_values('totalVotes', ascending=False)

	fig_kernel = go.Figure(data=[go.Table(
    header=dict(values=kernel_cols,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[d.head(100)[c] for c in kernel_cols],
               fill_color='lavender',
               align='left'))
	])

	return fig_kernel


if __name__ == '__main__':
    app.run_server(debug=True)

