import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go


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


competition_cols = ['competitionName', 'briefDescription', 'totalCompetitors', 'categories']
kernel_cols = ['competitionName', 'title',  'author', 'totalVotes', 'languageName', 'notebookFullUrl']





fig_kernel = go.Figure(data=[go.Table(
    header=dict(values=kernel_cols,
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_kernel.head(100)[c] for c in kernel_cols],
               fill_color='lavender',
               align='left'))
])



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.Div([
    	language_selector,
    	category_selector
    	]),


    dcc.Graph(id='fig_competition'),

    dcc.Graph(id='fig_kernel', figure=fig_kernel)
    
])


@app.callback(
	Output('fig_competition', 'figure'),
	[
	 Input('category', 'value')
	])
def update_competition_figure(category):
	# TODO: need fix
	d = df_competition[(df_competition.categories.isin(category))]
	print(category, d.shape)

	fig_competition = go.Figure(data=[go.Table(
	    header=dict(values=competition_cols,
	                fill_color='paleturquoise',
	                align='left'),
	    cells=dict(values=[d[c] for c in competition_cols],
	               fill_color='lavender',
	               align='left'))
	])

	return fig_competition



if __name__ == '__main__':
    app.run_server(debug=True)

