#Árbol de decisiones para la solución 

#Librerias
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz 
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

#cargar datos 
data = pd.read_excel("data/T.EN Colombia Reporte_Plano Lecciones Aprendidas.xlsx", sheet_name="confidencial")
print (data.head())

def limpiar(texto):
    nuevo_texto = texto.lower()
    nuevo_texto = re.sub('_x000d_', '', nuevo_texto)
    nuevo_texto = re.sub('&amp;quot', '', nuevo_texto)
    nuevo_texto = re.sub('quot', '', nuevo_texto)
    nuevo_texto = re.sub("\d+", '', nuevo_texto)
    nuevo_texto = re.sub("#N/D"," ",nuevo_texto)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , '', nuevo_texto)
    nuevo_texto = re.sub("\n+",' ',nuevo_texto)
    nuevo_texto = re.sub("  ", " ", nuevo_texto)
    
    return nuevo_texto

data["Tipo de contrato"] = data["Tipo de contrato"].astype(str)
data["Tipo de contrato"] = data["Tipo de contrato"].apply(limpiar)
data["Segmento Mercado"] = data["Segmento Mercado"].astype(str)
data["Segmento Mercado"]=data["Segmento Mercado"].apply(limpiar) 
data["Tipo de Alcance"] = data["Tipo de Alcance"].astype(str)
data["Tipo de Alcance"]=data["Tipo de Alcance"].apply(limpiar)
data["Tipo de Hallazgo"] = data["Tipo de Hallazgo"].astype(str)
data["Tipo de Hallazgo"]=data["Tipo de Hallazgo"].apply(limpiar)
data["Acciones Tomadas"] = data["Acciones Tomadas"].astype(str) 
data["Acciones Tomadas"]=data["Acciones Tomadas"].apply(limpiar)
data["Criticidad"] = data["Criticidad"].astype(str) 
data["Criticidad"]=data["Criticidad"].apply(limpiar)
data["Departamentos Responsables a Implementar"] = data["Departamentos Responsables a Implementar"].astype(str) 
data["Departamentos Responsables a Implementar"]=data["Departamentos Responsables a Implementar"].apply(limpiar) 
# Seleccionar las columnas relevantes para el modelo de árbol de decisión
columns = ['Tipo de contrato', 'Segmento Mercado', 'Tipo de Alcance',  'Acciones Tomadas', 'Causas', 'Tipo de Instalación','Departamentos Responsables a Implementar','Tema Ingenieria']
columnsV = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6','var7','var8']
# # Eliminar las filas que contengan valores nulos o faltantes
# data = data.dropna(subset=columns)
# Seleccionar las características (X) y las etiquetas (y)
X = data[columns]
#y = data['Estado']
y = data['Criticidad']
#y =data['Tipo de Hallazgo']
# Convertir características categóricas en variables dummy / codificación one-hot
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Crear el clasificador del árbol de decisión
clf = DecisionTreeClassifier(max_depth=5,criterion='entropy')

# Entrenar el modelo utilizando los datos de entrenamiento
clf.fit(X_train, y_train)

# Parameteros del modelo
print(clf.get_params())


# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)
print("Esta es la prediccion",y_pred[0])
# Obtener las características o caminos correspondientes a las predicciones
# Obtener el índice del nodo que lleva a la predicción para X_test[0]
indice_nodo = clf.decision_path(X_test[0:1]).indices

# Obtener las características y umbrales en el camino correcto para X_test[0]
caracteristicas_camino = []
for i, nodo in enumerate(indice_nodo):
    caracteristica = clf.tree_.feature[nodo]
    umbral = clf.tree_.threshold[nodo]
    caracteristicas_camino.append((caracteristica, umbral))

# Imprimir el camino desde la raíz hasta la predicción para X_test[0]
print("Camino desde la raíz hasta la predicción para X_test[0]:")
for caracteristica, umbral in caracteristicas_camino:
    nombre_caracteristica = X.columns[caracteristica]
    print(f"{nombre_caracteristica} <= {umbral}")



# Evaluar la precisión del modelo
print("Precisión:", metrics.accuracy_score(y_test, y_pred))

# # Matriz de confusión
# print(pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicción'], margins=True))

# Matriz de confusión

# print (confusion_matrix(y_test, y_pred, labels=[0,1]))

# Reporte de clasificación 
print (classification_report(y_test,y_pred))

#importancia de las características
# print(X.columns)
# print(clf.feature_importances_)

#Parte visual del arbol
# Generar el archivo DOT
# cantidad = len(X.columns)
# #generar nombres genericos para las caracteristicas
# nombresGenericos  = []
# indicador = 0
# for i in range(cantidad):
#     indicador = indicador + 1
#     nombresGenericos.append("Caracteristica "+ str(indicador) )

    
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True)

# Crear el objeto graphviz para visualizar el árbol
graph = graphviz.Source(dot_data)

# Guardar y visualizar el árbol en formato PDF
graph.format = 'pdf'
graph.render('Source', view=True)

# Guardar el archivo DOT para uso posterior
dot_file_path = Path('Source.gv')
dot_file_path.write_text(dot_data)


# # Crear una representación gráfica del árbol de decisión pero traba el computador
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
