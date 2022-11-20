import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Configuración de página
st.set_page_config(page_title="Sistema de predicción valor auto usado",page_icon="🚗",layout="centered")

# Para subir la animación del sidebar
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Configuración del sidebar
lottie_book = load_lottieurl('https://assets5.lottiefiles.com/private_files/lf30_skwgamub.json')
with st.sidebar:
    st.title('Trabajo Integrador Digital House')
    st.caption("Grupo 2: CRodriguez Geier & JSantacecilia")
    st.markdown("---")
my_button = st.sidebar.radio("Opciones",('Valuación autos', 'Parametros del trabajo'))
with st.sidebar:
    st.markdown("---")
    st_lottie(lottie_book, speed=0.5, height=200, key="initial")
    st.markdown("---")
    

#Configuración página principal
st.title("Digital Motors App")

#Llamando al modelo
def load_model():
    with open("./models/regressor.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
norm = data["normalization"]

le_manufacturer = LabelEncoder()
le_cylinder = LabelEncoder()
le_transmission = LabelEncoder()
le_drive = LabelEncoder()
le_paint_color = LabelEncoder()

def app():
    
    manufacturers = ('ford', 'gmc', 'chevrolet', 'toyota', 'jeep', 'nissan', 'honda',
       'dodge', 'chrysler', 'ram', 'mercedes-benz', 'bmw', 'volkswagen',
       'mazda', 'porsche', 'lexus', 'audi', 'mitsubishi', 'infiniti',
       'kia', 'hyundai', 'fiat', 'acura', 'cadillac', 'rover', 'lincoln',
       'jaguar', 'volvo', 'alfa-romeo', 'subaru', 'pontiac', 'saturn',
       'mini', 'buick', 'harley-davidson', 'mercury', 'datsun',
       'land rover', 'aston-martin', 'ferrari')
    cylinders = ('6 cylinders', '8 cylinders', '4 cylinders', '5 cylinders',
       '10 cylinders', '3 cylinders', '12 cylinders')
    transmissions = ('automatic', 'other', 'manual')
    drives = ('rwd', '4wd', 'fwd')
    paint_colors = ('black', 'silver', 'grey', 'red', 'blue', 'white', 'brown',
       'yellow', 'green', 'custom', 'orange', 'purple')
    
    form = st.form("my_form")
    manufacturer = form.selectbox('Marca: ', ["",manufacturers], format_func=lambda x: 'Elija una opción' if x == '' else x)
    year = form.number_input("Año: ")
    odometer = form.number_input("Kilometraje: ") 
    cylinder = form.selectbox('Cilindrada: ', ["", cylinders], format_func=lambda x: 'Elija una opción' if x == '' else x)
    transmission = form.selectbox('Transmisión: ', ["", transmissions], format_func=lambda x: 'Elija una opción' if x == '' else x)
    drive = form.selectbox('Tracción: ', ["", drives],format_func=lambda x: 'Elija una opción' if x == '' else x)
    paint_color = form.selectbox('Color: ', ["",paint_colors], format_func=lambda x: 'Elija una opción' if x == '' else x)
    submit = form.form_submit_button("Iniciar valuación")
    
    if submit:
        X = np.array([[manufacturer, year, odometer, cylinder, transmission, drive, paint_color]])
        X[:, 0] = le_manufacturer.fit_transform(X[:, 0])
        X[:, 3] = le_cylinder.fit_transform(X[:, 3])
        X[:, 4] = le_transmission.fit_transform(X[:, 4])
        X[:, 5] = le_drive.fit_transform(X[:, 5])
        X[:, 6] = le_paint_color.fit_transform(X[:, 6])
        scaled_X = norm.transform(X)

        price = model.predict(scaled_X)
        actual_price = np.exp(price) + 1
        actual_price = round(actual_price[0])

        st.subheader(f"La valuación del auto es: {actual_price:,} dolares")

