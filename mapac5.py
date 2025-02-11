# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:28:33 2025

@author: jperezr
"""


import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor



# Cambiar el fondo de la página a blanco
st.markdown("""
    <style>
        body {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Cambiar el fondo de los gráficos de Plotly
st.markdown("""
    <style>
        .js-plotly-plot {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)


# Diccionario de coordenadas de las 32 entidades federativas
estados_coords = {
    "Aguascalientes": [21.8853, -102.2916],
    "Baja California": [30.8406, -115.2838],
    "Baja California Sur": [26.0444, -111.6661],
    "Campeche": [19.8301, -90.5349],
    "Chiapas": [16.7569, -93.1292],
    "Chihuahua": [28.6330, -106.0691],
    "Ciudad de México": [19.4326, -99.1332],
    "Coahuila": [27.0587, -101.7068],
    "Colima": [19.2452, -103.7246],
    "Durango": [24.0277, -104.6530],
    "Guanajuato": [21.0190, -101.2574],
    "Guerrero": [17.4392, -99.5451],
    "Hidalgo": [20.0911, -98.7624],
    "Jalisco": [20.6595, -103.3494],
    "Estado de México": [19.4969, -99.7233],
    "Michoacán": [19.5665, -101.7068],
    "Morelos": [18.6813, -99.1013],
    "Nayarit": [21.7514, -104.8455],
    "Nuevo León": [25.5922, -99.9962],
    "Oaxaca": [17.0732, -96.7266],
    "Puebla": [19.0414, -98.2063],
    "Querétaro": [20.5888, -100.3899],
    "Quintana Roo": [19.1817, -88.4791],
    "San Luis Potosí": [22.1565, -100.9855],
    "Sinaloa": [25.1721, -107.4795],
    "Sonora": [29.2972, -110.3309],
    "Tabasco": [17.8409, -92.6189],
    "Tamaulipas": [24.2669, -98.8363],
    "Tlaxcala": [19.3133, -98.2404],
    "Veracruz": [19.1738, -96.1342],
    "Yucatán": [20.7099, -89.0943],
    "Zacatecas": [22.7709, -102.5832]
}

# Generar datos simulados para cada estado
np.random.seed(42)  # Para reproducibilidad
datos_estados = {}
for estado in estados_coords.keys():
    datos_estados[estado] = {
        "Población": np.random.randint(500000, 5000000),
        "Pensionados_Actuales": np.random.randint(10000, 200000),
        "Monto_Pensión": np.random.uniform(2000, 10000),
        "Inflación": np.random.uniform(3.0, 6.0),
        "Esperanza_Vida": np.random.uniform(70, 80),
        "Crecimiento_Poblacional": np.random.uniform(0.5, 2.5),
    }

# Función para proyectar pensionados usando árboles de decisión
def proyectar_pensionados(pensionados_actuales, crecimiento_poblacional, inflacion, años):
    X = np.array([crecimiento_poblacional, inflacion]).reshape(1, -1)
    model = DecisionTreeRegressor(random_state=42)
    # Datos de entrenamiento simulados
    X_train = np.random.rand(100, 2) * 10
    y_train = pensionados_actuales * (1 + X_train[:, 0]/100) ** años  # Simulación de crecimiento
    model.fit(X_train, y_train)
    return int(model.predict(X)[0])

# Título de la aplicación
st.title("Simulación y Proyección de Pensionados en México: Un Enfoque Basado en Datos")

# Sección de Ayuda
st.sidebar.title("Ayuda")
st.sidebar.write("""
Esta aplicación permite visualizar datos simulados de los estados de México, incluyendo proyecciones de pensionados utilizando un modelo de árboles de decisión.

- **Selecciona un estado** en el menú desplegable para ver su ubicación en el mapa y sus datos asociados.
- **Datos simulados**: Los datos de población, pensionados, inflación, etc., son generados aleatoriamente.
- **Proyecciones**: Se proyecta el número de pensionados a 5, 10 y 20 años utilizando un modelo de machine learning.
""")

# Método de Machine Learning utilizado
st.sidebar.title("Método de Machine Learning")
st.sidebar.write("""
El modelo utilizado para proyectar el número de pensionados es un **Árbol de Decisión para Regresión** (Decision Tree Regressor). Este modelo se entrena con datos simulados de crecimiento poblacional e inflación para predecir el número de pensionados en el futuro.
""")

# Desarrollado por
st.sidebar.title("Desarrollado por")
st.sidebar.write("Javier Horacio Pérez Ricárdez")

# Copyright
st.sidebar.title("Copyright")
st.sidebar.write("© 2023 Javier Horacio Pérez Ricárdez. Todos los derechos reservados.")

# Selección de estado
estado_seleccionado = st.selectbox("Selecciona un estado", list(estados_coords.keys()))

# Obtener coordenadas del estado seleccionado
lat, lon = estados_coords[estado_seleccionado]

# Crear DataFrame con la ubicación del estado
df_estado = pd.DataFrame({
    "Estado": [estado_seleccionado],
    "Latitud": [lat],
    "Longitud": [lon]
})



# Crear el mapa centrado en el estado seleccionado
fig = px.scatter_mapbox(
    df_estado,
    lat="Latitud",
    lon="Longitud",
    text="Estado",
    zoom=6,  # Ajuste de zoom inicial
    center={"lat": lat, "lon": lon},
    mapbox_style="carto-positron",  # Fondo oscuro con colores más vivos
    size=[20],  # Tamaño del marcador (más grande)
    color_discrete_sequence=["red"]  # Color rojo para el marcador
)

# Ajustar el layout del mapa para permitir zoom interactivo
fig.update_layout(
    mapbox=dict(
        style="carto-positron",  # Fondo oscuro con colores vivos
        zoom=6,
        center=dict(lat=lat, lon=lon)
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Eliminar márgenes
    height=600,  # Alto del mapa igual al ancho
    width=600    # Ancho del mapa
)

# Mostrar el mapa
st.plotly_chart(fig, use_container_width=True)




# Mostrar datos del estado seleccionado
st.subheader(f"Datos de {estado_seleccionado}")
datos_estado = datos_estados[estado_seleccionado]

# Proyectar pensionados a 5, 10 y 20 años
pensionados_actuales = datos_estado["Pensionados_Actuales"]
crecimiento_poblacional = datos_estado["Crecimiento_Poblacional"]
inflacion = datos_estado["Inflación"]

pensionados_5años = proyectar_pensionados(pensionados_actuales, crecimiento_poblacional, inflacion, 5)
pensionados_10años = proyectar_pensionados(pensionados_actuales, crecimiento_poblacional, inflacion, 10)
pensionados_20años = proyectar_pensionados(pensionados_actuales, crecimiento_poblacional, inflacion, 20)

# Crear DataFrame con los datos
df_datos_estado = pd.DataFrame({
    "Variable": [
        "Población", "Pensionados_Actuales", "Monto_Pensión", "Inflación", 
        "Esperanza_Vida", "Crecimiento_Poblacional", "Pensionados_5años", 
        "Pensionados_10años", "Pensionados_20años"
    ],
    "Valor": [
        datos_estado["Población"], datos_estado["Pensionados_Actuales"], 
        datos_estado["Monto_Pensión"], datos_estado["Inflación"], 
        datos_estado["Esperanza_Vida"], datos_estado["Crecimiento_Poblacional"], 
        pensionados_5años, pensionados_10años, pensionados_20años
    ]
})

# Mostrar el DataFrame
st.dataframe(df_datos_estado)



# Gráfico de barras de la proyección de pensionados
fig_barras = px.bar(
    x=["5 Años", "10 Años", "20 Años"],
    y=[pensionados_5años, pensionados_10años, pensionados_20años],
    labels={"x": "Años", "y": "Pensionados Proyectados"},
    title="Proyección de Pensionados en los Próximos Años"
)

# Formatear el hover para mostrar valores con separadores de miles
fig_barras.update_traces(
    hovertemplate="Pensionados Proyectados = %{y:,}<extra></extra>"
)

# Mostrar el gráfico de barras
st.plotly_chart(fig_barras, use_container_width=True)


# Gráfico de pastel de la distribución de pensionados actuales y proyectados
labels = ["Pensionados Actuales", "Proyección a 5 Años", "Proyección a 10 Años", "Proyección a 20 Años"]
values = [pensionados_actuales, pensionados_5años, pensionados_10años, pensionados_20años]

fig_pastel = px.pie(
    names=labels,
    values=values,
    title="Distribución de Pensionados Actuales y Proyectados"
)
st.plotly_chart(fig_pastel, use_container_width=True)