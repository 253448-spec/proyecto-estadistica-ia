import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai

# Configuración de página
st.set_page_config(
    page_title="Estadística con IA",
    layout="wide"
)

# Estilo simple (sin excesos)
st.markdown("""
    <style>
    .main-header {
        background-color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f9f9f9;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Análisis Estadístico con IA</h1>
    <p>Visualización de distribuciones | Prueba Z | Asistente Gemini</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 1. CARGA DE DATOS
# ============================================
st.sidebar.header("Panel de control")

opcion = st.sidebar.radio("Origen de datos", ["Subir CSV", "Generar datos sintéticos"])

datos = None
error_archivo = False
mensaje_error = ""

if opcion == "Subir CSV":
    archivo = st.sidebar.file_uploader("Selecciona un archivo CSV")
    
    if archivo:
        nombre = archivo.name
        extension = nombre.split('.')[-1].lower()
        
        if extension != "csv":
            error_archivo = True
            mensaje_error = f"Archivo rechazado: {nombre}\nExtensión detectada: .{extension}\nSolo se permiten archivos .csv"
        else:
            try:
                datos = pd.read_csv(archivo)
                st.sidebar.success(f"Archivo cargado: {nombre}")
            except Exception as e:
                error_archivo = True
                mensaje_error = f"Error al leer el archivo: {str(e)}"
    else:
        st.sidebar.info("Esperando archivo...")
else:
    st.sidebar.subheader("Configuración")
    n = st.sidebar.slider("Tamaño de muestra (n ≥ 30)", 30, 500, 100)
    tipo = st.sidebar.selectbox("Tipo de distribución", ["Normal", "Sesgada", "Con outliers"])
    
    if tipo == "Normal":
        datos = pd.DataFrame({"valor": np.random.normal(50, 10, n)})
    elif tipo == "Sesgada":
        datos = pd.DataFrame({"valor": np.random.gamma(2, 2, n)})
    else:
        datos = pd.DataFrame({"valor": np.concatenate([np.random.normal(50, 5, n-5), np.random.normal(80, 2, 5)])})
    st.sidebar.success(f"Generados {n} datos")

if error_archivo:
    st.error(mensaje_error)
    st.stop()

if datos is None:
    st.warning("Carga un archivo CSV o genera datos sintéticos para continuar")
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Vista previa de los datos")
st.dataframe(datos.head(), use_container_width=True)
st.caption(f"Dimensiones: {datos.shape[0]} filas x {datos.shape[1]} columnas")
st.markdown('</div>', unsafe_allow_html=True)

# Selección de variable numérica
num_cols = datos.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.error("No hay columnas numéricas en el archivo")
    st.stop()

variable = st.selectbox("Variable a analizar", num_cols)
vals = datos[variable].dropna()

# ============================================
# 2. VISUALIZACIONES
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Visualización de distribuciones")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(vals, kde=True, ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Histograma con curva de densidad")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(vals, patch_artist=True, boxprops=dict(facecolor="#95a5a6"))
    ax.set_title("Diagrama de caja")
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 3. INTERPRETACIÓN ESTADÍSTICA
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Interpretación automática")

media = vals.mean()
mediana = vals.median()
sesgo = vals.skew()
q75, q25 = vals.quantile(0.75), vals.quantile(0.25)
iqr = q75 - q25
outliers = ((vals < q25 - 1.5*iqr) | (vals > q75 + 1.5*iqr)).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Media", f"{media:.2f}")
col2.metric("Mediana", f"{mediana:.2f}")
col3.metric("Sesgo", f"{sesgo:.2f}")
col4.metric("Outliers", f"{outliers}")

if abs(sesgo) < 0.5:
    st.success("La distribución parece aproximadamente normal")
else:
    st.warning("La distribución NO parece normal (presenta sesgo significativo)")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 4. PRUEBA Z
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Prueba de hipótesis Z")
st.caption("Supuestos: varianza poblacional conocida, n ≥ 30")

media_muestral = vals.mean()
n_muestra = len(vals)

colA, colB = st.columns(2)
with colA:
    mu0 = st.number_input("Hipótesis nula (media)", value=round(media_muestral, 1), step=0.1)
    sigma = st.number_input("Desviación estándar poblacional", min_value=0.1, value=round(vals.std(), 1), step=0.1)
with colB:
    alpha = st.selectbox("Nivel de significancia", [0.01, 0.05, 0.10], index=1)
    cola = st.radio("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

z_calc = (media_muestral - mu0) / (sigma / np.sqrt(n_muestra))

if cola == "Bilateral":
    z_crit = stats.norm.ppf(1 - alpha/2)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z_calc)))
    rechazar = abs(z_calc) > z_crit
elif cola == "Cola izquierda":
    z_crit = stats.norm.ppf(alpha)
    p_valor = stats.norm.cdf(z_calc)
    rechazar = z_calc < z_crit
else:
    z_crit = stats.norm.ppf(1 - alpha)
    p_valor = 1 - stats.norm.cdf(z_calc)
    rechazar = z_calc > z_crit

st.subheader("Resultados de la prueba")

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
col_r1.metric("Estadístico Z", f"{z_calc:.4f}")
col_r2.metric("Valor crítico", f"{z_crit:.4f}")
col_r3.metric("p-value", f"{p_valor:.6f}")
col_r4.metric("Decisión", "Rechazar H0" if rechazar else "No rechazar H0")

# Curva de decisión
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax.plot(x, y, 'k-', linewidth=2)

if cola == "Bilateral":
    ax.fill_between(x, y, where=(x > z_crit) | (x < -z_crit), color="red", alpha=0.3)
    ax.axvline(-z_crit, color="red", linestyle=":", linewidth=2)
elif cola == "Cola izquierda":
    ax.fill_between(x, y, where=(x < z_crit), color="red", alpha=0.3)
else:
    ax.fill_between(x, y, where=(x > z_crit), color="red", alpha=0.3)

ax.axvline(z_calc, color="blue", linestyle="--", linewidth=2, label=f"Z = {z_calc:.2f}")
ax.axvline(z_crit, color="red", linestyle=":", linewidth=2, label=f"Z crítico = {z_crit:.2f}")
ax.legend()
ax.set_title("Curva de decisión (región de rechazo en rojo)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 5. ASISTENTE IA (GEMINI)
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Asistente IA (Gemini)")

api_key = st.text_input("API Key de Google Gemini", type="password")

if api_key:
    genai.configure(api_key=api_key)
    modelo_ia = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
Se realizó una prueba Z con los siguientes parámetros:
media muestral = {media_muestral:.2f}, media hipotética = {mu0}, n = {n_muestra}, sigma = {sigma}, alpha = {alpha}, tipo de prueba = {cola}.

El estadístico Z calculado fue = {z_calc:.4f}
El p-value fue = {p_valor:.6f}
La decisión automática es: {'Rechazar H₀' if rechazar else 'No rechazar H₀'}

Preguntas:
1. ¿Se rechaza H₀? Explica la decisión y si los supuestos de la prueba son razonables.
2. Según tu análisis, ¿la decisión automática es correcta?
3. ¿La IA cometería algún error en este análisis? Explica.
"""
    
    if st.button("Consultar a Gemini", use_container_width=True):
        with st.spinner("Gemini está analizando..."):
            respuesta = modelo_ia.generate_content(prompt)
            
            st.markdown("### Respuesta de Gemini")
            st.info(respuesta.text)
            
            # Comparación entre IA y app
            st.markdown("---")
            st.markdown("### Comparación: IA vs Decisión automática")
            
            texto_ia = respuesta.text.lower()
            
            col_ia, col_app = st.columns(2)
            
            with col_ia:
                st.markdown("**Decisión de la IA**")
                if "rechazar" in texto_ia and "no rechazar" not in texto_ia:
                    decision_ia = "Rechazar H₀"
                    color_ia = "green"
                elif "no rechazar" in texto_ia:
                    decision_ia = "No rechazar H₀"
                    color_ia = "red"
                else:
                    decision_ia = "No se pudo determinar"
                    color_ia = "orange"
                st.markdown(f"<h3 style='color:{color_ia}'>{decision_ia}</h3>", unsafe_allow_html=True)
            
            with col_app:
                st.markdown("**Decisión de la aplicación**")
                decision_app = "Rechazar H₀" if rechazar else "No rechazar H₀"
                color_app = "green" if rechazar else "red"
                st.markdown(f"<h3 style='color:{color_app}'>{decision_app}</h3>", unsafe_allow_html=True)
            
            if ("rechazar" in texto_ia and rechazar) or ("no rechazar" in texto_ia and not rechazar):
                st.success("Coincidencia: La IA y la aplicación están de acuerdo")
            else:
                st.warning("Discrepancia: La IA y la aplicación no están de acuerdo. Revisar manualmente.")
else:
    st.info("Ingresa tu API Key para activar el asistente")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Proyecto de Estadística con IA - Documentación del proceso creativo")