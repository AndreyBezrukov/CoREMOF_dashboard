# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:42:26 2020
@author: Andrey.Bezrukov
"""

import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# file location and name
filepath = '2019-11-01-ASR-public_12020.csv'

# import data and clean
df = pd.read_csv(filepath)

df['disorder'] = df.DISORDER.apply(lambda x: 'Yes' if x=='DISORDER' else 'No')
df.LCD = df.LCD.round(2)
df.PLD = df.PLD.round(2)

axis_label_dict = {
            'LCD':u'Largest Cavity Diameter, Å', 
            'PLD':u'Pore Limiting Diameter, Å', 
            'LFPD':u'Largest Cavity Diameter, Å', 
            'cm3_g':'Density, cm3/g', 
            'ASA_m2_cm3':'Available Surface Area, m2/cm3' ,
            'ASA_m2_g':'Available Surface Area, m2/g',
            'NASA_m2_cm3':'Not Available Surface Area, m2/cm3', 
            'NASA_m2_g':'Not Available Surface Area, m2/g',
            'AV_VF':'Available Volume Fraction', 
            'AV_cm3_g':'Available Volume cm3/g', 
            'NAV_cm3_g':'Not Available Volume cm3/g'
            }

#print(sorted(list(set( [item for sublist in df['All_Metals'].values for item in sublist.split(',')] ))))

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.LCD, y=df.PLD,
                         text=df.filename, mode='markers',
                         marker=dict(line_width=1,colorscale='Viridis', showscale=True, color=df.ASA_m2_g, colorbar=dict(title='Available Surface Area, m2/g'),),
                         )),
fig.update_layout(
xaxis = dict(
        title_text = u"Largest Cavity Diameter, Å",
        title_font = {"size": 20},),
yaxis = dict(
        title_text = u"Pore Limiting Diameter, Å",
        title_font = {"size": 20}
    ))

app.layout = html.Div(children=[
    html.H1(children='CoREMOF database dashboard'),
    html.Div(children='''
        Explore 12000+ Computation-Ready, Experimental Metal−Organic Framework structures deposited in the database.  
    '''),
    html.Label([html.A('CoREMOF 2019 reference', href='https://pubs.acs.org/doi/pdf/10.1021/acs.jced.9b00835')]),
    html.Label('X Axis:'),
    dcc.Dropdown(
        id='xaxis_selection',
        options=[
                {'label': 'Largest Cavity Diameter', 'value': 'LCD'  }  ,
                {'label': 'Pore Limiting Diameter', 'value': 'PLD'  }  ,
                {'label': 'Largest Cavity Diameter', 'value': 'LFPD'  }  ,
                {'label': 'Density', 'value': 'cm3_g'  }  ,
                {'label': 'Available Surface Area m2/cm3', 'value': 'ASA_m2_cm3'  }  ,
                {'label': 'Available Surface Area m2/g', 'value': 'ASA_m2_g'  }  ,
                {'label': 'Not Available Surface Area m2/cm3', 'value': 'NASA_m2_cm3'  }  ,
                {'label': 'Not Available Surface Area m2/g', 'value': 'NASA_m2_g'  }  ,
                {'label': 'Available Volume Fraction', 'value': 'AV_VF'  }  ,
                {'label': 'Available Volume cm3/g', 'value': 'AV_cm3_g'  }  ,
                {'label': 'Not Available Volume cm3/g', 'value': 'NAV_cm3_g'  }  ,
        ],
        value='LCD'
    ),
    html.Label('Y Axis:'),
    dcc.Dropdown(
        id='yaxis_selection',
        options=[
                {'label': 'Largest Cavity Diameter', 'value': 'LCD'  }  ,
                {'label': 'Pore Limiting Diameter', 'value': 'PLD'  }  ,
                {'label': 'Largest Cavity Diameter', 'value': 'LFPD'  }  ,
                {'label': 'Density', 'value': 'cm3_g'  }  ,
                {'label': 'Available Surface Area m2/cm3', 'value': 'ASA_m2_cm3'  }  ,
                {'label': 'Available Surface Area m2/g', 'value': 'ASA_m2_g'  }  ,
                {'label': 'Not Available Surface Area m2/cm3', 'value': 'NASA_m2_cm3'  }  ,
                {'label': 'Not Available Surface Area m2/g', 'value': 'NASA_m2_g'  }  ,
                {'label': 'Available Volume Fraction', 'value': 'AV_VF'  }  ,
                {'label': 'Available Volume cm3/g', 'value': 'AV_cm3_g'  }  ,
                {'label': 'Not Available Volume cm3/g', 'value': 'NAV_cm3_g'  }  ,
        ],
        value='PLD'
    ),
    html.Label('Z Axis:'),
    dcc.Dropdown(
        id='zaxis_selection',
        options=[
                {'label': 'Largest Cavity Diameter', 'value': 'LCD'  }  ,
                {'label': 'Pore Limiting Diameter', 'value': 'PLD'  }  ,
                {'label': 'Largest Cavity Diameter', 'value': 'LFPD'  }  ,
                {'label': 'Density', 'value': 'cm3_g'  }  ,
                {'label': 'Available Surface Area m2/cm3', 'value': 'ASA_m2_cm3'  }  ,
                {'label': 'Available Surface Area m2/g', 'value': 'ASA_m2_g'  }  ,
                {'label': 'Not Available Surface Area m2/cm3', 'value': 'NASA_m2_cm3'  }  ,
                {'label': 'Not Available Surface Area m2/g', 'value': 'NASA_m2_g'  }  ,
                {'label': 'Available Volume Fraction', 'value': 'AV_VF'  }  ,
                {'label': 'Available Volume cm3/g', 'value': 'AV_cm3_g'  }  ,
                {'label': 'Not Available Volume cm3/g', 'value': 'NAV_cm3_g'  }  ,
        ],
        value='ASA_m2_g'
    ),
    html.Label('Metal in the structure:'),
    dcc.Dropdown(
        id='all_metal-dropdown',
        options=[
                {'label': 'Ag', 'value': 'Ag'  }  ,
                {'label': 'Al', 'value': 'Al'  }  ,
                {'label': 'As', 'value': 'As'  }  ,
                {'label': 'Au', 'value': 'Au'  }  ,
                {'label': 'Ba', 'value': 'Ba'  }  ,
                {'label': 'Be', 'value': 'Be'  }  ,
                {'label': 'Bi', 'value': 'Bi'  }  ,
                {'label': 'Ca', 'value': 'Ca'  }  ,
                {'label': 'Cd', 'value': 'Cd'  }  ,
                {'label': 'Ce', 'value': 'Ce'  }  ,
                {'label': 'Co', 'value': 'Co'  }  ,
                {'label': 'Cr', 'value': 'Cr'  }  ,
                {'label': 'Cs', 'value': 'Cs'  }  ,
                {'label': 'Cu', 'value': 'Cu'  }  ,
                {'label': 'Dy', 'value': 'Dy'  }  ,
                {'label': 'Er', 'value': 'Er'  }  ,
                {'label': 'Eu', 'value': 'Eu'  }  ,
                {'label': 'Fe', 'value': 'Fe'  }  ,
                {'label': 'Ga', 'value': 'Ga'  }  ,
                {'label': 'Gd', 'value': 'Gd'  }  ,
                {'label': 'Ge', 'value': 'Ge'  }  ,
                {'label': 'Hf', 'value': 'Hf'  }  ,
                {'label': 'Hg', 'value': 'Hg'  }  ,
                {'label': 'Ho', 'value': 'Ho'  }  ,
                {'label': 'In', 'value': 'In'  }  ,
                {'label': 'Ir', 'value': 'Ir'  }  ,
                {'label': 'K', 'value': 'K'    },
                {'label': 'La', 'value': 'La'  }  ,
                {'label': 'Li', 'value': 'Li'  }  ,
                {'label': 'Lu', 'value': 'Lu'  }  ,
                {'label': 'Mg', 'value': 'Mg'  }  ,
                {'label': 'Mn', 'value': 'Mn'  }  ,
                {'label': 'Mo', 'value': 'Mo'  }  ,
                {'label': 'Na', 'value': 'Na'  }  ,
                {'label': 'Nb', 'value': 'Nb'  }  ,
                {'label': 'Nd', 'value': 'Nd'  }  ,
                {'label': 'Ni', 'value': 'Ni'  }  ,
                {'label': 'Np', 'value': 'Np'  }  ,
                {'label': 'Pb', 'value': 'Pb'  }  ,
                {'label': 'Pd', 'value': 'Pd'  }  ,
                {'label': 'Pr', 'value': 'Pr'  }  ,
                {'label': 'Pt', 'value': 'Pt'  }  ,
                {'label': 'Pu', 'value': 'Pu'  }  ,
                {'label': 'Rb', 'value': 'Rb'  }  ,
                {'label': 'Re', 'value': 'Re'  }  ,
                {'label': 'Rh', 'value': 'Rh'  }  ,
                {'label': 'Ru', 'value': 'Ru'  }  ,
                {'label': 'Sb', 'value': 'Sb'  }  ,
                {'label': 'Sc', 'value': 'Sc'  }  ,
                {'label': 'Si', 'value': 'Si'  }  ,
                {'label': 'Sm', 'value': 'Sm'  }  ,
                {'label': 'Sn', 'value': 'Sn'  }  ,
                {'label': 'Sr', 'value': 'Sr'  }  ,
                {'label': 'Tb', 'value': 'Tb'  }  ,
                {'label': 'Te', 'value': 'Te'  }  ,
                {'label': 'Th', 'value': 'Th'  }  ,
                {'label': 'Ti', 'value': 'Ti'  }  ,
                {'label': 'Tm', 'value': 'Tm'  }  ,
                {'label': 'U', 'value': 'U'    },
                {'label': 'V', 'value': 'V'    },
                {'label': 'W', 'value': 'W'    },
                {'label': 'Y', 'value': 'Y'    },
                {'label': 'Yb', 'value': 'Yb'  }  ,
                {'label': 'Zn', 'value': 'Zn'  }  ,
                {'label': 'Zr', 'value': 'Zr'  }  ,
        ],
        multi=True,
        value=['Ag', 'Al', 'As', 'Au', 'Ba', 'Be', 
               'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 
               'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'Fe', 
               'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 
               'In', 'Ir', 'K', 'La', 'Li', 'Lu', 
               'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 
               'Ni', 'Np', 'Pb', 'Pd', 'Pr', 'Pt', 
               'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'Sb', 
               'Sc', 'Si', 'Sm', 'Sn', 'Sr', 'Tb', 
               'Te', 'Th', 'Ti', 'Tm', 'U',  'V', 
               'W', 'Y', 'Yb', 'Zn', 'Zr']
    ),
    html.Label('Disorder:'),
    dcc.Checklist(
        id='dis-checkbox',
        options=[
            {'label': 'has disorder', 'value': 'Yes'},
            {'label': 'no disorder', 'value': 'No'},
        ],
        value=['Yes', 'No']
    ),
    html.Label('Open Metal Site (OMS):'),
    dcc.Checklist(
        id='oms-checkbox',
        options=[
            {'label': 'has OMS', 'value': 'Yes'},
            {'label': 'no OMS', 'value': 'No'},
        ],
        value=['Yes', 'No']
    ),
    
    dcc.Graph(
        id='graph',
        figure=fig
    ),
    
    html.Label('Z axis range slider'),
    dcc.RangeSlider(
        id='Z-slider',
        min=df.ASA_m2_g.min(),
        max=df.ASA_m2_g.max(),
        allowCross=False,
        value=[df.ASA_m2_g.min(), df.ASA_m2_g.max()],
        marks=dict(zip([df.ASA_m2_g.min()+j/10*(df.ASA_m2_g.max()-df.ASA_m2_g.min()) for j in range(10)], [{'label':str(round(i))} for i in [df.ASA_m2_g.min()+j/10*(df.ASA_m2_g.max()-df.ASA_m2_g.min()) for j in range(10)]]))
    )
])

@app.callback( 
    [Output(component_id='Z-slider', component_property='min'),
     Output(component_id='Z-slider', component_property='max'),
     Output(component_id='Z-slider', component_property='marks'),
     Output(component_id='Z-slider', component_property='step'),
    ],
    [Input('zaxis_selection', 'value'),]
    )
def fill_second_slider(selected_zaxis):
     return df[selected_zaxis].min(), df[selected_zaxis].max(), dict(zip([df[selected_zaxis].min()+j/10*(df[selected_zaxis].max()-df[selected_zaxis].min()) for j in range(10)], [{'label':str(round(i, 1))} for i in [df[selected_zaxis].min()+j/10*(df[selected_zaxis].max()-df[selected_zaxis].min()) for j in range(10)]])), (df[selected_zaxis].max()-df[selected_zaxis].min())/300

@app.callback(
    Output('graph', 'figure'),
     
    [
    Input('xaxis_selection', 'value'),
    Input('yaxis_selection', 'value'),
    Input('zaxis_selection', 'value'),
    Input('all_metal-dropdown', 'value'),
    Input('Z-slider', 'value'),
    Input('dis-checkbox', 'value'),
    Input('oms-checkbox', 'value'),
    ])
def update_figure(selected_xaxis, selected_yaxis, selected_zaxis, selected_all_metal, selected_Z, selected_dis, selected_oms ):
    filtered_df = df[([any([item for item in sublist.split(',') if item in selected_all_metal]) for sublist in df['All_Metals'].values])&
                     (df[selected_zaxis] >= selected_Z[0])&
                     (df[selected_zaxis] <= selected_Z[1])&
                     (df.disorder.isin(selected_dis))&
                     (df.Has_OMS.isin(selected_oms))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df.LCD, y=filtered_df.PLD,
                         text=filtered_df.filename, mode='markers',
                         marker=dict(line_width=1,colorscale='Viridis', showscale=True, color=filtered_df[selected_zaxis], colorbar=dict(title=axis_label_dict[selected_zaxis]),),
                         )),
    scatter = fig.data[0]
    scatter.x = filtered_df[selected_xaxis]
    scatter.y = filtered_df[selected_yaxis]

    fig.update_layout(
    xaxis = dict(
        title_text = axis_label_dict[selected_xaxis],
        title_font = {"size": 20},),
    yaxis = dict(
        title_text = axis_label_dict[selected_yaxis],
        title_font = {"size": 20}
    ))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
