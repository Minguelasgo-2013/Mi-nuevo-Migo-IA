# 🤖 Mi Gemini Desde Cero

Un modelo de lenguaje tipo Transformer construido desde cero con PyTorch,
entrenado a nivel de carácter con un dataset de instrucciones en español.

## Estructura del proyecto

- `modelo.py` — arquitectura del Transformer (bloques de atención causal)
- `tokenizador.py` — tokenizador a nivel de carácter
- `convertir_datos.py` — convierte el dataset JSON en texto de entrenamiento limpio
- `main.py` — entrena el modelo
- `generar.py` — chatea con el modelo por consola
- `app.py` — interfaz web con Streamlit

## Cómo usarlo localmente

```bash
pip install -r requirements.txt

# 1. Convertir los datos (necesitas alpaca_data_cleaned_spanish.json en la carpeta)
python convertir_datos.py

# 2. Entrenar
python main.py

# 3. Probar por consola
python generar.py

# 4. O lanzar la interfaz web
streamlit run app.py
```

## Despliegue en línea

Este proyecto está pensado para desplegarse gratis en
[Streamlit Community Cloud](https://share.streamlit.io), conectando
directamente este repositorio de GitHub.

## Nota

Es un proyecto educativo para aprender cómo funciona un Transformer por dentro,
no un modelo de producción — las respuestas reflejan el tamaño (deliberadamente
pequeño) del modelo y del entrenamiento.