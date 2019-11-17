# -*- coding: utf-8 -*-
"""
Creates Interactive Scatter plot based raw data
To look at distributions of two metrics at a time
"""
import os
import pickle
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from tqdm import tqdm
from load_data import load_parse_save

def u_pickle_origin_csv(pkl_file='./parsed_dataframe.pkl'):
  """
  Grabs either pickled csv or processes a new one & picke/return it
  """
  # Load Data from csv file & process via load_parse_save
  if os.path.isfile(pkl_file):
    # Alternative is Load from Pickled set: Saves Time
    with open(pkl_file, 'rb') as fp:
      ret_dict = pickle.load(fp)
  else:
    print("Pkl FileNotFound - so the text will be processed. This may take time")
    tqdm.pandas(desc='load & parse csv')
    ret_dict = load_parse_save(save_file='./parsed_dataframe.pkl', debug=True)

  prev_df = ret_dict['csv_df']      #: csv parse & load
  # days prior to max & min
  return prev_df

# grab pickled csv or process csv
base_df = u_pickle_origin_csv()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# create RadioItems Options
listofobjs = [{'label': 'metric{}'.format(x), 'value': 'metric{}'.format(x)} \
              for x in range(1, 10)]

app.layout = html.Div(
  children=[
    html.H1(children='Metrics comparison Frequency in Dash'),
    html.Div(children='''Try choosing the two different radio buttons Top & Bottom'''),
    # Here's Interactive dropdown
    html.Label('Radio Choice Top'),
    dcc.RadioItems(
      id='RChoice1',
      options=listofobjs,
      value='metric3',
      labelStyle={'display': 'inline-block'}
    ),

    html.Label('Radio Choice Bottom'),
    dcc.RadioItems(
      id='RChoice2',
      options=listofobjs,
      value='metric1',
      labelStyle={'display': 'inline-block'}
    ),

    dcc.Graph(
      id='topViolin',
      figure={
        'data': [
          go.Violin(x=base_df['Fail_set'][base_df['Fail_set'] == fail_set],
                    y=base_df['metric3'][base_df['Fail_set'] == fail_set],
                    name="{}".format(fail_set),
                    box_visible=True,
                    meanline_visible=True,)
          for fail_set in [True, False]
        ]
      }
    ),

    dcc.Graph(
      id='bottomViolin',
      figure={
        'data': [
          go.Violin(x=base_df['Fail_set'][base_df['Fail_set'] == fail_set],
                    y=base_df['metric1'][base_df['Fail_set'] == fail_set],
                    name="{}".format(fail_set),
                    box_visible=True,
                    meanline_visible=True,)
          for fail_set in [True, False]
        ]
      }
    ),


  ], style={'colCount':2}
)

@app.callback(
    [Output('topViolin', 'figure')],
    [Input('RChoice1', 'value')],
)

def update_graphics1(RChoice1): # clickData):
  """
  Updates the top graph based on 1st RadioItems
  """
  # Let's do something with clicks later
  #if clickData is not None:
  traces = [
    go.Violin(x=base_df['Fail_set'][base_df['Fail_set'] == fail_set],
              y=base_df[RChoice1][base_df['Fail_set'] == fail_set],
              name="{}".format(fail_set),
              box_visible=True,
              meanline_visible=True,)
    for fail_set in [True, False]
  ]
  return [{'data': traces}]

@app.callback(
    [Output('bottomViolin', 'figure')],
    [Input('RChoice2', 'value')],
)

def update_graphics2(RChoice2):
  """
  Updates the bottom graph based on 2nd RadioItems
  """
  traces = [
    go.Violin(x=base_df['Fail_set'][base_df['Fail_set'] == fail_set],
              y=base_df[RChoice2][base_df['Fail_set'] == fail_set],
              name="{}".format(fail_set),
              box_visible=True,
              meanline_visible=True,)
    for fail_set in [True, False]
  ]
  return [{'data': traces}]

if __name__ == '__main__':
  app.run_server(debug=True)
