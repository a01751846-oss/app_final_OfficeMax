import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
st.set_page_config(page_title="App Pricing & Elasticidad", layout="wide")
COLOR_PALETTE = ['#1f77b4', '#9467bd', '#00b1b2', '#2ca02c', '#17becf']

COSTO2_ES_UNITARIO = True
ELIMINAR_COSTO_MAYOR_O_IGUAL_PRECIO = False
UMBRAL_CV_VAR_ALTA = 2.0
UMBRAL_REGISTROS_REMOVIDOS_AMARILLO = 0.25
UMBRAL_REGISTROS_REMOVIDOS_ROJO = 0.50

ESCENARIOS_CAMBIO_PRECIO = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
ESCENARIOS_PROMOCION = [
    {"Nombre_Escenario": "Promoción 2x1", "Cambio_Efectivo": -0.50},
    {"Nombre_Escenario": "Promoción 3x2", "Cambio_Efectivo": -1/3},
    {"Nombre_Escenario": "Promoción 2do a 50%", "Cambio_Efectivo": -0.25},
]
ESCENARIOS_COMPLETOS = [{"Nombre_Escenario": f"Cambio de precio {c*100:+.0f}%", "Cambio_Efectivo": c} for c in ESCENARIOS_CAMBIO_PRECIO] + ESCENARIOS_PROMOCION

# ==========================================
# FUNCIONES MATEMÁTICAS DEL NOTEBOOK
# ==========================================
def preparar_df_modelo(df):
    """Agrega ventas por día y precio."""
    cols_req = ["fecha_dia", "precio_modelo", "qty", "net_sale"]
    df_m = df.dropna(subset=cols_req).copy()
    df_m = df_m[(df_m["qty"] > 0) & (df_m["net_sale"] > 0) & (df_m["precio_modelo"] > 0)]
    if df_m.empty: return pd.DataFrame(columns=["qty_modelo", "precio_modelo"])
    
    df_agg = df_m.groupby(["fecha_dia", "precio_modelo"], as_index=False).agg(
        qty_modelo=("qty", "sum"), venta_modelo=("net_sale", "sum")
    )
    df_agg["precio_modelo"] = df_agg["venta_modelo"] / df_agg["qty_modelo"]
    return df_agg[(df_agg["qty_modelo"] > 0) & (df_agg["precio_modelo"] > 0)].copy()

def estimar_elasticidad_loglog(df):
    """Cálculo real de elasticidad usando Statsmodels OLS."""
    df_modelo = preparar_df_modelo(df)
    n_modelo = len(df_modelo)
    precios_distintos = df_modelo["precio_modelo"].nunique() if n_modelo > 0 else 0
    
    if n_modelo < 3 or precios_distintos < 2:
        return np.nan, np.nan, np.nan, np.nan, n_modelo, "Datos insuficientes"
        
    df_modelo["log_qty"] = np.log(df_modelo["qty_modelo"])
    df_modelo["log_precio"] = np.log(df_modelo["precio_modelo"])
    
    X = df_modelo["log_precio"]
    y = df_modelo["log_qty"]
    X = sm.add_constant(X)
    
    try:
        modelo = sm.OLS(y, X).fit()
        beta = modelo.params["log_precio"]
        alfa = modelo.params["const"]
        r2 = modelo.rsquared
        p_value = modelo.pvalues["log_precio"]
        
        diag = "OK" if beta < 0 else "Elasticidad Positiva"
        return beta, alfa, r2, p_value, n_modelo, diag
    except:
        return np.nan, np.nan, np.nan, np.nan, n_modelo, "Error Statsmodels"

# ==========================================
# LIMPIEZA Y CRUCE DE BASES
# ==========================================
@st.cache_data
def procesar_bases(df_ventas, df_promos, df_nse):
    """Lógica de limpieza, márgenes y cruce con NSE."""
    v = df_ventas.copy()
    v.columns = v.columns.str.strip()
    filas_orig = len(v)
    
    # Transformaciones base
    v["tran_date"] = pd.to_datetime(v["tran_date"], errors="coerce")
    for col in ["qty", "net_sale", "costo2"]:
        v[col] = pd.to_numeric(v[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    
    v = v.dropna(subset=["tran_date", "qty", "net_sale", "costo2", "prod_nbr"])
    v = v[(v["qty"] > 0) & (v["net_sale"] > 0)]
    
    # Precio y Costo
    v["precio_unitario"] = v["net_sale"] / v["qty"]
    v["costo_unitario"] = v["costo2"] if COSTO2_ES_UNITARIO else v["costo2"] / v["qty"]
    v = v[(v["precio_unitario"] > 0) & (v["costo_unitario"] >= 0)]
    
    if ELIMINAR_COSTO_MAYOR_O_IGUAL_PRECIO:
        v = v[v["costo_unitario"] < v["precio_unitario"]]
        
    # Variables de negocio
    v["ingreso_base"] = v["precio_unitario"] * v["qty"]
    v["margen_unitario"] = v["precio_unitario"] - v["costo_unitario"]
    v["margen_total"] = v["margen_unitario"] * v["qty"]
    
    # Variables de tiempo y modelo
    v["trimestre"] = v["tran_date"].dt.to_period("Q").astype(str)
    v["mes"] = v["tran_date"].dt.to_period("M").astype(str)
    v["fecha_dia"] = v["tran_date"].dt.date
    v["precio_modelo"] = v["precio_unitario"].round(2)
    
    # Cruce NSE (Simulando la lógica geográfica)
    if df_nse is not None and "id_municipio" in v.columns and "ubica_geo" in df_nse.columns:
        v["id_municipio"] = v["id_municipio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        df_nse["ubica_geo"] = df_nse["ubica_geo"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        
        est_socio_municipio = df_nse.dropna(subset=["ubica_geo", "est_socio"]).groupby("ubica_geo")["est_socio"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        v = v.merge(est_socio_municipio.rename(columns={"ubica_geo": "id_municipio", "est_socio": "est_socio_nbr"}), on="id_municipio", how="left")
        mapa_nse = {1: "Bajo", 2: "Medio Bajo", 3: "Medio Alto", 4: "Alto", "1": "Bajo", "2": "Medio Bajo", "3": "Medio Alto", "4": "Alto"}
        v["categoria_est_socio"] = v["est_socio_nbr"].map(mapa_nse)
    else:
        v["categoria_est_socio"] = "Sin Clasificación"
        
    # KPIs de calidad
    removidos = filas_orig - len(v)
    calidad = {"limpias": len(v), "removidos": removidos, "pct": (removidos/filas_orig)*100, "orig": filas_orig}
    
    if calidad["pct"] >= UMBRAL_REGISTROS_REMOVIDOS_ROJO*100: calidad["semaforo"] = "🔴 Rojo"
    elif calidad["pct"] >= UMBRAL_REGISTROS_REMOVIDOS_AMARILLO*100: calidad["semaforo"] = "🟡 Amarillo"
    else: calidad["semaforo"] = "🟢 Verde"
    
    return v, calidad

# ==========================================
# BARRA LATERAL (SIDEBAR) & NAVEGACIÓN
# ==========================================
st.sidebar.title("Navegación y Carga")
vista = st.sidebar.radio("Ir a vista:", ["1. Carga y Diagnóstico", "2. Elasticidad", "3. Pricing Dinámico"])

st.sidebar.markdown("---")
ventas_file = st.sidebar.file_uploader("1. Ventas (Obligatorio)", type=['csv', 'xlsx'], help="Archivo histórico necesario con columnas: tran_date, qty, net_sale, prod_nbr, costo2.")
promos_file = st.sidebar.file_uploader("2. Promociones (Opcional)", type=['csv', 'xlsx'], help="Base con el calendario de promociones.")
nse_file = st.sidebar.file_uploader("3. NSE (Opcional)", type=['csv', 'xlsx'], help="Base INEGI (hogares_INEGI.csv).")

if ventas_file:
    df_raw = pd.read_csv(ventas_file) if ventas_file.name.endswith('.csv') else pd.read_excel(ventas_file)
    df_promos = pd.read_csv(promos_file) if promos_file and promos_file.name.endswith('.csv') else (pd.read_excel(promos_file) if promos_file else None)
    df_nse = pd.read_csv(nse_file) if nse_file and nse_file.name.endswith('.csv') else (pd.read_excel(nse_file) if nse_file else None)
    
    df, stats_calidad = procesar_bases(df_raw, df_promos, df_nse)
else:
    df = None
    st.sidebar.warning("Sube la base de ventas para comenzar.")

# ==========================================
# VISTA 1: CARGA Y DIAGNÓSTICO
# ==========================================
if vista == "1. Carga y Diagnóstico" and df is not None:
    st.title("Carga y Diagnóstico de Datos")
    
    # Semáforo
    if "Verde" in stats_calidad["semaforo"]: st.success(f"✅ Calidad Aceptable: Base lista para modelar.")
    elif "Amarillo" in stats_calidad["semaforo"]: st.warning(f"⚠️ Calidad Regular: Se removió el {stats_calidad['pct']:.1f}% de los datos.")
    else: st.error(f"❌ Calidad Deficiente: Se removieron demasiados registros.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros Originales", f"{stats_calidad['orig']:,}")
    col2.metric("Registros Eliminados", f"{stats_calidad['removidos']:,}")
    col3.metric("Registros Limpios", f"{stats_calidad['limpias']:,}")
    col4.metric("SKUs Únicos", f"{df['prod_nbr'].nunique():,}")

    st.markdown("---")
    st.subheader("Base Limpia + Nivel Socioeconómico")
    st.dataframe(df.head(100), use_container_width=True)
    st.info("ℹ️ La tabla muestra la base cruzada con 'categoria_est_socio' y márgenes unitarios.")

# ==========================================
# VISTA 2: ELASTICIDAD
# ==========================================
elif vista == "2. Elasticidad" and df is not None:
    st.title("Modelo de Elasticidad Trimestral OLS")
    st.markdown("Dashboard interactivo para visualizar la sensibilidad al precio por SKU según su modelo log-log.")
    
    col1, col2, col3 = st.columns(3)
    deptos = ["Todos"] + list(df["dept_nm"].dropna().unique()) if "dept_nm" in df.columns else ["Todos"]
    depto_sel = col1.selectbox("Departamento ℹ️", deptos, help="Filtro jerárquico.")
    
    df_f1 = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    trimestres = ["Todos"] + list(df_f1["trimestre"].dropna().unique())
    trim_sel = col2.selectbox("Trimestre ℹ️", trimestres, help="Periodo para el modelo de elasticidad.")
    
    df_f2 = df_f1[df_f1["trimestre"] == trim_sel] if trim_sel != "Todos" else df_f1
    skus = list(df_f2["prod_nbr"].dropna().unique())
    sku_sel = col3.selectbox("SKU ℹ️", skus, help="Producto específico para ejecutar la regresión OLS.")

    if sku_sel:
        df_sku = df_f2[df_f2["prod_nbr"] == sku_sel].copy()
        beta, alfa, r2, p_value, obs, diag = estimar_elasticidad_loglog(df_sku)
        
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elasticidad (Beta)", f"{beta:.3f}" if pd.notna(beta) else "N/A")
        k2.metric("R²", f"{r2:.2f}" if pd.notna(r2) else "N/A")
        k3.metric("Observaciones Válidas", obs)
        k4.metric("Diagnóstico", diag)
        
        # Gráficas
        g1, g2, g3 = st.columns(3)
        with g1:
            if obs > 2:
                fig1 = px.scatter(preparar_df_modelo(df_sku), x="precio_modelo", y="qty_modelo", trendline="ols", title="Curva de Elasticidad OLS", color_discrete_sequence=[COLOR_PALETTE[0]])
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("Ajuste de mínimos cuadrados ordinarios. Demanda vs Precio.")
        with g2:
            fig2 = px.line(df_sku.groupby("tran_date")["qty"].sum().reset_index(), x="tran_date", y="qty", title="Demanda en el Tiempo", color_discrete_sequence=[COLOR_PALETTE[1]])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Comportamiento histórico de unidades vendidas.")
        with g3:
            if "estado" in df_sku.columns:
                fig3 = px.bar(df_sku.groupby("estado")["qty"].sum().reset_index(), x="estado", y="qty", title="Demanda por Estado", color_discrete_sequence=[COLOR_PALETTE[2]])
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Mapa de calor / barras de demanda regional.")
            else:
                st.info("Sin columna geográfica 'estado' para mapa.")
        
        st.download_button("📥 Descargar CSV Elasticidad", df_sku.to_csv(index=False).encode('utf-8'), "elasticidad_detalle.csv", "text/csv")

# ==========================================
# VISTA 3: PRICING DINÁMICO
# ==========================================
elif vista == "3. Pricing Dinámico" and df is not None:
    st.title("Pricing Dinámico y Proyecciones")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    trim_sel = col1.selectbox("Trimestre", list(df["trimestre"].dropna().unique()))
    dept_sel = col2.selectbox("Departamento", ["Todos"] + list(df["dept_nm"].dropna().unique()) if "dept_nm" in df.columns else ["Todos"])
    df_f = df[(df["trimestre"] == trim_sel)]
    if dept_sel != "Todos": df_f = df_f[df_f["dept_nm"] == dept_sel]
    
    nse_sel = col3.selectbox("NSE", ["Todos"] + list(df_f["categoria_est_socio"].dropna().unique()))
    if nse_sel != "Todos": df_f = df_f[df_f["categoria_est_socio"] == nse_sel]
    
    sku_sel = col4.selectbox("SKU (Unidad)", list(df_f["prod_nbr"].unique()))
    escenario_sel = col5.selectbox("Escenario a simular", [e["Nombre_Escenario"] for e in ESCENARIOS_COMPLETOS])
    
    if sku_sel:
        # Extraer parámetros y correr modelo base
        df_sku = df_f[df_f["prod_nbr"] == sku_sel]
        beta, _, _, _, _, _ = estimar_elasticidad_loglog(df_sku)
        
        # Simulación
        cambio_efectivo = next(e["Cambio_Efectivo"] for e in ESCENARIOS_COMPLETOS if e["Nombre_Escenario"] == escenario_sel)
        elasticidad_usada = np.clip(beta, -5, 0) if pd.notna(beta) and beta < 0 else -1.0 # Fallback si es positivo o nulo
        
        unidades_base = df_sku["qty"].sum()
        precio_base = df_sku["precio_unitario"].mean()
        costo_base = df_sku["costo_unitario"].mean()
        
        ingreso_base = precio_base * unidades_base
        margen_base = (precio_base - costo_base) * unidades_base
        
        # Ecuación notebook: unidades_simuladas = unidades_base * np.exp(elasticidad_usada * np.log1p(cambio))
        precio_nuevo = precio_base * (1 + cambio_efectivo)
        unidades_sim = unidades_base * np.exp(elasticidad_usada * np.log1p(cambio_efectivo))
        ingreso_sim = precio_nuevo * unidades_sim
        margen_sim = (precio_nuevo - costo_base) * unidades_sim
        
        st.markdown("---")
        st.subheader("Impacto Financiero del Escenario")
        k1, k2, k3 = st.columns(3)
        k1.metric("Unidades Proyectadas", f"{unidades_sim:,.0f}", f"{unidades_sim - unidades_base:,.0f} vs Base")
        k2.metric("Ingreso Proyectado", f"${ingreso_sim:,.2f}", f"${ingreso_sim - ingreso_base:,.2f} vs Base")
        k3.metric("Margen Proyectado", f"${margen_sim:,.2f}", f"${margen_sim - margen_base:,.2f} vs Base")
        
        g1, g2, g3 = st.columns(3)
        with g1:
            fig1 = go.Figure(data=[go.Bar(name='Base', x=['Ventas $'], y=[ingreso_base], marker_color='gray'), go.Bar(name='Escenario', x=['Ventas $'], y=[ingreso_sim], marker_color=COLOR_PALETTE[0])])
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Comparación de Ventas (Ingreso) Base vs Escenario.")
        with g2:
            fig2 = go.Figure(data=[go.Bar(name='Base', x=['Unidades'], y=[unidades_base], marker_color='gray'), go.Bar(name='Escenario', x=['Unidades'], y=[unidades_sim], marker_color=COLOR_PALETTE[2])])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Comparación de Volumen de Unidades.")
        with g3:
            fig3 = go.Figure(data=[go.Bar(name='Base', x=['Margen $'], y=[margen_base], marker_color='gray'), go.Bar(name='Escenario', x=['Margen $'], y=[margen_sim], marker_color=COLOR_PALETTE[3])])
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Impacto en rentabilidad neta (Margen).")

        st.info(f"💡 **Conclusión:** Para el SKU {sku_sel}, aplicar un {escenario_sel} genera un cambio de {((margen_sim/margen_base)-1)*100:.1f}% en el margen total bajo una elasticidad calculada de {elasticidad_usada:.2f}.")
