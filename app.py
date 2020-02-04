import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go
import pickle
import numpy as np

#df = pd.read_csv('data_dash.csv')
df = pd.read_csv('C:/Users/Sergio/Desktop/Python/data_dash.csv')


dff = df.drop(columns="left")
with open('C:/Users/Sergio/Desktop/Python/predicciones.pkl', 'rb') as file:
#with open('predicciones.pkl', 'rb') as file:   
            model = pickle.load(file)


score=model.predict_proba(dff)



dff= pd.DataFrame(score, columns=['Abandona', 'No_abandona'])


df=pd.merge(df, dff, left_index=True, right_index=True)


df = df.drop(columns="No_abandona")

pd.set_option('display.max_columns', 500)



app = dash.Dash(__name__)

Departamentos= df[[ 'IT', 'RandD', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical']]

Departamentos= Departamentos.stack().reset_index()

Departamentos=Departamentos.loc[Departamentos[0]==1].reset_index(drop=True)

Departamentos=Departamentos['level_1']

df=df.merge(Departamentos, left_index=True, right_index=True)

df=df[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years','level_1', 'salary', 'Abandona']]

df.columns=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
       'promotion_last_5years', 'Departamentos','salary', 'Abandona']



df.columns =['Nivel de satisfacción', 'Última evaluación', 'Número de proyectos',
 'Horas mensuales trabajadas', 'Años en la compañía', 'Accidente laboral', 'Abandono', 'Ascenso en los últimos cinco años', 'Departamento',
  'Nivel de salario', 'Predicción del abandono']


df2=df.drop(["Abandono", 'Predicción del abandono'],1)


params= ['Nivel de satisfacción', 'Última evaluación', 'Número de proyectos',
 'Horas mensuales trabajadas', 'Años en la compañía', 'Accidente laboral', 'Ascenso en los últimos cinco años', 'Departamento',
  'Nivel de salario']



df.loc[(df.Departamento == 'IT'),'Departamento']='Tecnología de la información'
df.loc[(df.Departamento == 'RandD'),'Departamento']='Investigación y desarrollo'
df.loc[(df.Departamento == 'accounting'),'Departamento']='Contabilidad'
df.loc[(df.Departamento == 'hr'),'Departamento']='Recursos humanos'
df.loc[(df.Departamento == 'management'),'Departamento']='Administración'
df.loc[(df.Departamento == 'marketing'),'Departamento']='Marketing'
df.loc[(df.Departamento == 'product_mng'),'Departamento']='Producción'
df.loc[(df.Departamento == 'sales'),'Departamento']='Ventas'
df.loc[(df.Departamento == 'support'),'Departamento']='Soporte'
df.loc[(df.Departamento == 'technical'),'Departamento']='Técnico'

df.loc[(df.Abandono ==0),'Abandono']='No'
df.loc[(df.Abandono ==1),'Abandono']='Si'


df.loc[(df['Ascenso en los últimos cinco años'] ==0),'Ascenso en los últimos cinco años']='No'
df.loc[(df['Ascenso en los últimos cinco años'] ==1),'Ascenso en los últimos cinco años']='Si'

df.loc[(df['Accidente laboral'] ==0),'Accidente laboral']='No'
df.loc[(df['Accidente laboral'] ==1),'Accidente laboral']='Si'

df.loc[(df['Nivel de salario'] ==3),'Nivel de salario']='Alto'
df.loc[(df['Nivel de salario'] ==1),'Nivel de salario']='Bajo'
df.loc[(df['Nivel de salario'] ==2),'Nivel de salario']='Medio'

df['Predicción del abandono'] = df['Predicción del abandono'].apply(lambda x: x*100)
df['Predicción del abandono'] = df['Predicción del abandono'].apply( lambda x : str(int(x)) + '%')





app.layout = html.Div([
    html.Div(html.H1('PANEL DE CONTROL DE RRHH'), style = {"fontFamily": "Open Sans",'color':'rgb(0, 64, 128)'}),
    html.Div([
        html.Div([
                html.Div(html.H2('Departamentos'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px'}),
                dcc.Dropdown(
                        id='selector_departamento',
                        options=[{'label': i, 'value': i} for i in df.Departamento.unique()],
                        searchable = False,
                        value=None
                        ) 
            ],style = {'margin':'10px', 'width':'40%', 'align':'left','backgroundColor': 'rgb(250, 250, 250)','padding': '10px 5px'}),
                
            html.Div([
                html.Div(html.H2('Abandono'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px', 'margin-right':'5px'}),
                dcc.RadioItems(
                        id='selector_abandono',
                        options=[{'label': i, 'value': i} for i in ['Si', 'No', 'Global']],
                        labelStyle={'display': 'block',  'fontSize':'22px'},
                        value = 'Global'
                        )
            ], style = {'margin':'10px', 'width':'20%','backgroundColor': 'rgb(250, 250, 250)','padding': '10px 5px'}),


            html.Div([
                html.Div(html.H2('Accidente laboral'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px', 'margin-right':'5px'}),
                dcc.RadioItems(
                        id='selector_accidente',
                        options=[{'label': i, 'value': i} for i in ['Si', 'No', 'Global']],
                        labelStyle={'display': 'block',  'fontSize':'22px'},
                        value = 'Global'
                        )
            ], style = {'margin':'10px', 'width':'20%','backgroundColor': 'rgb(250, 250, 250)','padding': '10px 5px'}),

            html.Div([
                html.Div(html.H2('Ascenso en los últimos cinco años'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px', 'margin-right':'5px'}),
                dcc.RadioItems(
                        id='selector_ascenso',
                        options=[{'label': i, 'value': i} for i in ['Si', 'No', 'Global']],
                        labelStyle={'display': 'block',  'fontSize':'22px'},
                        value = 'Global'
                        )
            ], style = {'margin':'10px', 'width':'20%','backgroundColor': 'rgb(250, 250, 250)','padding': '10px 5px'})


    ], style ={ 'display':'inline-flex','width':'97%','borderBottom': 'thin lightgrey solid','padding-bottom':'10px', 'borderTop': 'thin lightgrey solid', 'backgroundColor': 'rgb(250, 250, 250)'}),



    html.Div([
        html.Div([html.H1('Información de los empleados')], style = {"fontFamily": "Open Sans",'padding-left':'20px','textAlign':'left','color':'rgb(0, 64, 128)'}),
        dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        sort_action='native',
        style_table={
        'maxHeight': '450px',
        'overflowY': 'scroll'},
        style_cell = {"fontFamily": "Arial", 'textAlign':'center', "size": 10},
        
        style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "0%"'
            },
            'backgroundColor': '#208000',
            'color': 'white',
        },
        
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "10%"'
            },
            'backgroundColor': '#408000',
            'color': 'white',
        },

        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "20%"'
            },
            'backgroundColor': '#608000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "30%"'
            },
            'backgroundColor': '#808000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "40%"'
            },
            'backgroundColor': '#808000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "50%"'
            },
            'backgroundColor': '#806000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "60%"'
            },
            'backgroundColor': '#804000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "70%"'
            },
            'backgroundColor': '#804000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "80%"'
            },
            'backgroundColor': '#802000',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "90%"'
            },
            'backgroundColor': '#800000',
            'color': 'white',
        },
        
        {
            'if': {
                'column_id': 'Predicción del abandono',
                'filter_query': '{Predicción del abandono} eq "100%"'
            },
            'backgroundColor': '#800000',
            'color': 'white',
        }

        ]





        )
    ], style={'display': 'inline-block', 'width': '97%', 'textAlign': 'center','borderBottom': 'thin lightgrey solid', 'backgroundColor':'rgb(250, 250, 250)','border-radius': '5px'},
    ),


    html.Div([
        html.Div([html.H1('Predicción del abandono')], style = {"fontFamily": "Open Sans",'padding-left':'20px','textAlign':'left','color':'rgb(0, 64, 128)'}),
            html.Div([
                html.Div([
                dash_table.DataTable(
                id='tabla2',
                columns=[{"name": i, "id": i} for i in df2.columns],
                data=[  dict({param: None for param in params})],
                editable=True,
                style_cell = {"fontFamily": "Arial", 'textAlign':'center', "size": 10})
                ]), 

                html.Div(
                id= 'Resultado',
                style = {'textAlign':'left', 'color': 'rgb(0, 64, 128)',"fontSize": "25px", 'padding':'40px 0px 40px 20px'})
            ])

    ], style={'display': 'inline-block', 'width': '97%', 'textAlign': 'center','borderBottom': 'thin lightgrey solid', 'border-radius': '5px','backgroundColor':'rgb(250, 250, 250)'}
    ),     
 


    html.Div([

        html.Div([
            html.Div([
                    html.Div(html.H2('Histogramas de las variables'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px'}),
                    html.Div(html.H3('Variable representada en el histograma:'), style ={ "fontFamily": "sans-serif", 'padding-left':'15px','textAlign':'left', "size": '15px','color':'rgb(0, 64, 128)', 'margin-top':'40px'}),
                    html.Div([
                        dcc.Dropdown(
                                id='selector_histogramas',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                searchable = False,
                                value='Nivel de satisfacción'
                                ) 
                    ],style={'width':'50%'})

                ],style = {'margin':'10px 10px 10px 10px', 'padding': '10px 5px'}),


            html.Div([
                dcc.Graph(
                    id='grafo_satis'
                    )                
            ], style={'margin':'120px 25px 25px 25px', 'padding': '0 20'})
        ], style={'width': '50%',  'margin':'25px', 'padding': '0 20'}),

        html.Div([
            html.Div([
                html.Div(html.H2('Diagrama de dispersión'), style ={"fontFamily": "Open Sans",'backgroundColor':'rgb(153, 204, 255)', 'border-radius': '5px'}),
                html.Div([
                    html.Div([

                        html.Div(html.H3('Variable del eje X:'), style ={ 'textAlign':'left',"fontFamily": "sans-serif",'padding-left':'15px',"size": '15px','color':'rgb(0, 64, 128)'}),
                        dcc.Dropdown(
                                id='selector_dispersion1',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                searchable = False,
                                value='Nivel de satisfacción'
                                ),
                        
                        html.Div(html.H3('Variable del eje Y:'), style ={'textAlign':'left', "fontFamily": "sans-serif",'padding-left':'15px',"size": '15px','color':'rgb(0, 64, 128)'}),
                        dcc.Dropdown(
                                id='selector_dispersion2',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                searchable = False,
                                value='Última evaluación'
                                )
                    ], style = {'width': '50%'}),
                    html.Div([
                        html.Div(html.H3('Abandono:'), style ={"fontFamily": "sans-serif",'padding-left':'40px','textAlign':'left', "size": '15px','color':'rgb(0, 64, 128)'}),
                        dcc.RadioItems(
                            id='selector_dispersion3',
                            options=[{'label': i, 'value': i} for i in ['No mostrar el abandono', 'Mostrar según el abandono']],
                            labelStyle={"fontFamily": "Open Sans",'fontSize':'22px', 'padding-left':'40px', 'margin-bottom':'20px', 'display': 'block', 'textAlign':'left'},
                            value = 'No mostrar el abandono')
                    ], style={'width': '50%'})
                ], style={'display':'flex'})
                                 
                    
        ],style = {'margin':'10px 10px 10px 30px',  'padding': '10px 5px'}),

            html.Div([
                dcc.Graph(
                    id='grafo_disper'                    
                )
            ], style={'margin':'25px','padding': '0 20'})
        ],style={'width': '50%', 'margin':'25px', 'padding': '0 20'})

    ], style={'width': '100%','display': 'inline-flex'})





],style={'width': '80%', 'textAlign': 'center', 'margin-left':'10%', "border":"20px rgb(204, 229, 255) groove", 'border-radius':'30px'})




@app.callback(
    Output(component_id='Resultado', component_property='children'),
    [Input('tabla2', 'data'),
     Input('tabla2', 'columns')])

def predictor(rows, columns):

    dfaux = pd.DataFrame(rows, columns=[c['name'] for c in columns])


    
    if dfaux['Nivel de satisfacción'].iloc[0] is not None:
        try: 
            float(dfaux['Nivel de satisfacción'])
        except ValueError:
            return 'Introduzca un valor válido en la variable "Nivel de satisfacción".'
    
        if (float(dfaux['Nivel de satisfacción']) > 1.0) or (float(dfaux['Nivel de satisfacción']) < 0.0):
            return 'Introduzca un valor decimal entre 0.00 y 1.00 en la variable "Nivel de satisfacción".'

    
    
    if dfaux['Última evaluación'].iloc[0] is not None:
        try: 
            float(dfaux['Última evaluación'])
        except ValueError:
            return 'Introduzca un valor válido en la variable "Última evaluación".'

        if (float(dfaux['Última evaluación']) > 1.0) or (float(dfaux['Última evaluación']) < 0.0):
            return 'Introduzca un valor decimal entre 0.00 y 1.00 en la variable "Última evaluación".'


    if dfaux['Número de proyectos'].iloc[0] is not None:
        try: 
            int(dfaux['Número de proyectos'])
        except ValueError:
            return 'Introduzca un valor válido en la variable "Número de proyectos".'

        if (int(dfaux['Número de proyectos']) > 10) or (int(dfaux['Número de proyectos']) < 0):
            return 'Introduzca un valor entre 0 y 10 en la variable "Número de proyectos".'


    if dfaux['Horas mensuales trabajadas'].iloc[0] is not None:
        try: 
            int(dfaux['Horas mensuales trabajadas'])
        except ValueError:
            return 'Introduzca un valor válido en la variable "Horas mensuales trabajadas".'
        
        if (int(dfaux['Horas mensuales trabajadas']) > 350) or (int(dfaux['Horas mensuales trabajadas']) < 90):
            return 'Introduzca un valor entre 90 y 350 en la variable "Horas mensuales trabajadas".'
            



    if dfaux['Años en la compañía'].iloc[0] is not None:
        try: 
            int(dfaux['Años en la compañía'])
        except ValueError:
            return 'Introduzca un valor válido en la variable "Años en la compañía".' 

        if (int(dfaux['Años en la compañía']) > 10) or (int(dfaux['Años en la compañía']) < 0):
            return 'Introduzca un valor entre 0 y 10 en la variable "Años en la compañía".'
    
    
    if dfaux['Accidente laboral'].iloc[0] is not None:
      
        if (dfaux['Accidente laboral'].iloc[0].lower() != 'si') and (dfaux['Accidente laboral'].iloc[0].lower() != 'no'):
            return 'Introduzca "si" o "no" en la variable "Accidente laboral".'

    if dfaux['Ascenso en los últimos cinco años'].iloc[0] is not None:
      
        if (dfaux['Ascenso en los últimos cinco años'].iloc[0].lower() != 'si') and (dfaux['Ascenso en los últimos cinco años'].iloc[0].lower() != 'no'):
            return 'Introduzca "si" o "no" en la variable "Ascenso en los últimos cinco años".'

    


    if dfaux['Departamento'].iloc[0] is not None:
      
        if (dfaux['Departamento'].iloc[0].lower() != 'tecnología de la información') and \
            (dfaux['Departamento'].iloc[0].lower() != 'investigación y desarrollo') and \
            (dfaux['Departamento'].iloc[0].lower() != 'contabilidad') and \
            (dfaux['Departamento'].iloc[0].lower() != 'recursos humanos') and \
            (dfaux['Departamento'].iloc[0].lower() != 'administración') and \
            (dfaux['Departamento'].iloc[0].lower() != 'marketing') and \
            (dfaux['Departamento'].iloc[0].lower() != 'producción') and \
            (dfaux['Departamento'].iloc[0].lower() != 'ventas') and \
            (dfaux['Departamento'].iloc[0].lower() != 'soporte') and \
            (dfaux['Departamento'].iloc[0].lower() != 'técnico'):
            return 'Introduzca el nombre de alguno de los departamentos en la variable "Departamento".'


    if dfaux['Nivel de salario'].iloc[0] is not None:
      
        if (dfaux['Nivel de salario'].iloc[0].lower() != 'bajo') and \
            (dfaux['Nivel de salario'].iloc[0].lower() != 'medio') and \
            (dfaux['Nivel de salario'].iloc[0].lower() != 'alto'):
            return 'Introduzca un nivel "bajo", "medio" o "alto" en la variable "Nivel de salario".'

        
    if dfaux['Nivel de satisfacción'].iloc[0] is not None and dfaux['Última evaluación'].iloc[0] is not None and\
        dfaux['Número de proyectos'].iloc[0] is not None and dfaux['Horas mensuales trabajadas'].iloc[0] is not None and\
        dfaux['Años en la compañía'].iloc[0] is not None and dfaux['Accidente laboral'].iloc[0] is not None and\
        dfaux['Ascenso en los últimos cinco años'].iloc[0] is not None and dfaux['Departamento'].iloc[0] is not None and\
        dfaux['Nivel de salario'].iloc[0] is not None:
        

        dfaux=dfaux.applymap(str.lower)

        dfaux['Ascenso en los últimos cinco años']=np.where(dfaux['Ascenso en los últimos cinco años']=='si', 1,0)

        dfaux['Accidente laboral']=np.where(dfaux['Accidente laboral']=='si', 1,0)

        
        
        dfaux.loc[(dfaux['Nivel de salario'] =='bajo'),'Nivel de salario']= 1
        dfaux.loc[(dfaux['Nivel de salario'] =='medio'),'Nivel de salario']= 2
        dfaux.loc[(dfaux['Nivel de salario'] =='alto'),'Nivel de salario']= 3



        dfaux['tecnología de la información']=0
        dfaux['investigación y desarrollo']=0
        dfaux['contabilidad']=0
        dfaux['recursos humanos']=0
        dfaux['administración']=0
        dfaux['marketing']=0
        dfaux['producción']=0
        dfaux['ventas']=0
        dfaux['soporte']=0
        dfaux['técnico']=0

        dfaux[dfaux['Departamento'].iloc[0]]=1

        dfaux=dfaux.drop(columns=['Departamento'],axis=1)


        
        dfaux.loc[(dfaux['Nivel de salario']== 'bajo'),'Nivel de salario']=1
        dfaux.loc[(dfaux['Nivel de salario'] == 'medio'),'Nivel de salario']=2
        dfaux.loc[(dfaux['Nivel de salario' ]== 'alto'),'Nivel de salario']=3

        
        
        score=model.predict_proba(dfaux)
        dat= pd.DataFrame(score, columns=['Abandona', 'No_abandona'])

        resultado = dat['Abandona'].values[0]
    
        resultado=resultado*100

        

        return  'El empleado con las caracteristicas introducidas tiene una probabilidad de abandonar la compañía del {}%.'.format(int(resultado))




@app.callback(
    dash.dependencies.Output('table', 'data'),
    [dash.dependencies.Input('selector_departamento', 'value'),
     dash.dependencies.Input('selector_abandono', 'value'),
     dash.dependencies.Input('selector_accidente', 'value'),
     dash.dependencies.Input('selector_ascenso', 'value')])
def update_table(valor_departamento, valor_abandono, valor_accidente, valor_ascenso):

    dff = df[df['Departamento'] == valor_departamento]

    if valor_departamento is None:
        dff = df
        
    if valor_abandono == 'No':
        dff = dff[dff['Abandono'] == 'No']
    
    if valor_abandono == 'Si':
        dff = dff[dff['Abandono'] == 'Si']

    if valor_accidente == 'No':
        dff = dff[dff['Accidente laboral'] == 'No']
    
    if valor_accidente == 'Si':
        dff = dff[dff['Accidente laboral'] == 'Si']

    if valor_ascenso == 'No':
        dff = dff[dff['Ascenso en los últimos cinco años'] == 'No']
    
    if valor_ascenso == 'Si':
        dff = dff[dff['Ascenso en los últimos cinco años'] == 'Si']


    return dff.to_dict('records')




@app.callback(
    dash.dependencies.Output('grafo_satis', 'figure'),
    [dash.dependencies.Input('selector_departamento', 'value'),
     dash.dependencies.Input('selector_abandono', 'value'),
     dash.dependencies.Input('selector_accidente', 'value'),
     dash.dependencies.Input('selector_ascenso', 'value'),
     dash.dependencies.Input('selector_histogramas', 'value')])
def update_graf1(valor_departamento, valor_abandono, valor_accidente, valor_ascenso, valor_histograma):

    dff = df[df['Departamento'] == valor_departamento]

    if valor_departamento is None:
        dff = df
        
    if valor_abandono == 'No':
        dff = dff[dff['Abandono'] == 'No']
    
    if valor_abandono == 'Si':
        dff = dff[dff['Abandono'] == 'Si']

    if valor_accidente == 'No':
        dff = dff[dff['Accidente laboral'] == 'No']
    
    if valor_accidente == 'Si':
        dff = dff[dff['Accidente laboral'] == 'Si']

    if valor_ascenso == 'No':
        dff = dff[dff['Ascenso en los últimos cinco años'] == 'No']
    
    if valor_ascenso == 'Si':
        dff = dff[dff['Ascenso en los últimos cinco años'] == 'Si']


    return {'data':[go.Histogram(
                x= dff[valor_histograma],
                marker=dict(
                        color='rgb(128, 191, 255)'
                    )
                )],
                'layout':dict(
                title=dict(text ='Distribución de la variable "{}"'.format(valor_histograma),
                               font =dict(
                               size=20,
                               color = 'rgb(0, 64, 128)')),
                margin=dict(l=40, r=0, t=100, b=30)
                )}



@app.callback(
    dash.dependencies.Output('grafo_disper', 'figure'),
    [dash.dependencies.Input('selector_dispersion1', 'value'),
     dash.dependencies.Input('selector_dispersion2', 'value'),
     dash.dependencies.Input('selector_dispersion3', 'value')])
     
def update_graf2(valor_x, valor_y, valor_abandono):


    if valor_x is None:
        return {
        'data': [],
        'layout': dict(
            xaxis={
                'title': valor_x
            },
            yaxis={
                'title': valor_y
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450)
        }

    if valor_y is None:
        return {
        'data': [],
        'layout': dict(
            xaxis={
                'title': valor_x
            },
            yaxis={
                'title': valor_y
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450)
        }

    if valor_abandono =='Mostrar según el abandono':
        traces=[]
        dfabs=df[df['Abandono']=='Si']
        dfabn=df[df['Abandono']=='No']

        traces.append(dict(
            x=dfabs[valor_x],
            y=dfabs[valor_y],
            text=dfabs['Abandono'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.1,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Si'
        ))


        
        traces.append(dict(
            x=dfabn[valor_x],
            y=dfabn[valor_y],
            text=dfabn['Abandono'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.1,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='No'
        ))

        return {
            'data': traces,
            'layout': dict(
                xaxis={
                    'title': valor_x
                },
                yaxis={
                    'title': valor_y
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450
            )
            }



    return {
        'data': [dict(  
            x=df[valor_x],
            y=df[valor_y],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.1,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': dict(
            xaxis={
                'title': valor_x
            },
            yaxis={
                'title': valor_y
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450
        )
    }






if __name__ == '__main__':
    app.run_server(debug=True)