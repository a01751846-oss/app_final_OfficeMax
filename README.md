# 📊 App de Pricing Dinámico y Elasticidad

Esta aplicación interactiva construida en **Streamlit** permite realizar diagnósticos de bases de datos de ventas, calcular la elasticidad precio de la demanda mediante modelos OLS (Log-Log) y proyectar escenarios de pricing dinámico para maximizar el ingreso y margen.

## 🚀 Características Principales
1. **Carga y Diagnóstico:** Limpieza automática de datos, cruce geográfico con bases de INEGI (NSE) y semáforo de calidad basado en la varianza.
2. **Modelo de Elasticidad:** Regresión lineal (Statsmodels) por SKU y trimestre para entender la sensibilidad al precio.
3. **Pricing Dinámico:** Simulación de escenarios promocionales y de precios, categorización de SKUs y recomendación del escenario ideal para maximizar margen/volumen.

## 🛠️ Instalación y Uso
1. Clona este repositorio: `git clone <tu_url_aqui>`
2. Crea tu entorno virtual: `python -m venv venv`
3. Activa el entorno e instala dependencias: `pip install -r requirements.txt`
4. Ejecuta la aplicación: `streamlit run app.py`
