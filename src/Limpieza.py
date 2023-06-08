#Limpieza de datos 

import pandas as pd 
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

data = pd.read_excel("data/T.EN Colombia Reporte_Plano Lecciones Aprendidas.xlsx", sheet_name="confidencial")
data['Fecha'] = pd.to_datetime(data['Fecha'])

casos_anuales = data.groupby(data['Fecha'].dt.year).size()

fig = go.Figure(data=[go.Bar(x=casos_anuales.index, y=casos_anuales)])
fig.update_layout(title='Casos Anuales', xaxis_title='Año', yaxis_title='Cantidad de Casos')
fig.write_image("fecha.png")

# Gráfico de cantidad de casos por empleado
casos_empleado = data['Codigo Empleado'].value_counts().reset_index()
casos_empleado.columns = ['Empleado', 'Cantidad de Casos']



fig = px.scatter(casos_empleado, x="Empleado", y="Cantidad de Casos")
fig.write_image("empleados.png", engine='kaleido', width=1200, height=900)


fig = px.bar(data, x='Estado', color = "Tipo de Hallazgo")
fig.write_image("estado.png", engine='kaleido', width=1200, height=900)


fig = px.bar(data, x='Segmento Mercado', color="Criticidad")
fig.write_image("mercado.png", engine='kaleido', width=1200, height=900)



data['Fecha de inicio'] = pd.to_datetime(data['Fecha de inicio'], errors='coerce')
data['Fecha de fin'] = pd.to_datetime(data['Fecha de fin'], errors='coerce')

data['DiasTranscurridos'] = (data['Fecha de fin'] - data['Fecha de inicio']).dt.days
print(data)