#Árbol de decisiones para la solución de la opsción 5

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Cargar datos
pqrs = pd.read_excel("/home/humath/Desktop/AndreaOT/CAOBA/Datos/DATA_PQR_2022 21032023.xlsx", sheet_name="PQRSD 2022")

# Eliminar filas con valores nulos en las columnas relevantes
pqrs = pqrs.dropna(subset=["DESCRIPCION_HECHOS", "Modo", "Motivo", "MEDIO_RECEPCION", "TIPO_PQR"])

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(pqrs["DESCRIPCION_HECHOS"], pqrs["TIPO_PQR"], test_size=0.2, random_state=42)

# Crear una instancia del vectorizador con las stopwords en español
vectorizer = CountVectorizer(stop_words="spanish", max_features=1000)

# Ajustar el vectorizador al conjunto de entrenamiento y transformar los datos
X_train = vectorizer.fit_transform(X_train)

# Transformar los datos de prueba utilizando el vectorizador ajustado en el conjunto de entrenamiento
X_test = vectorizer.transform(X_test)

# Crear el modelo de árbol de decisión
clf = DecisionTreeClassifier()

# Ajustar el modelo a los datos de entrenamiento
clf.fit(X_train, y_train)

# Realizar la predicción en los datos de prueba
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
print(classification_report(y_test, y_pred))