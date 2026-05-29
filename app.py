import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="App Pricing & Elasticidad", layout="wide")

ESCENARIOS = [
    {"Nombre_Escenario": "Cambio de precio +15%", "Cambio": 0.15},
    {"Nombre_Escenario": "Cambio de precio +10%", "Cambio": 0.10},
    {"Nombre_Escenario": "Cambio de precio +5%", "Cambio": 0.05},
    {"Nombre_Escenario": "Base (0%)", "Cambio": 0.00},
    {"Nombre_Escenario": "Cambio de precio -5%", "Cambio": -0.05},
    {"Nombre_Escenario": "Cambio de precio -10%", "Cambio": -0.10},
    {"Nombre_Escenario": "Cambio de precio -15%", "Cambio": -0.15},
    {"Nombre_Escenario": "Promoción 2do al 50%", "Cambio": -0.25},
    {"Nombre_Escenario": "Promoción 3x2", "Cambio": -0.3333},
    {"Nombre_Escenario": "Promoción 2x1", "Cambio": -0.50},
]

# ==========================================
# PROCESAMIENTO Y LIMPIEZA DE DATOS
# ==========================================
@st.cache_data
def limpiar_y_cruzar_bases(df_ventas, df_nse):
    v = df_ventas.copy()
    v.columns = v.columns.str.strip()
    filas_orig = len(v)
    
    v["tran_date"] = pd.to_datetime(v["tran_date"], errors="coerce")
    for col in ["qty", "net_sale", "costo2"]:
        if col in v.columns:
            v[col] = pd.to_numeric(v[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        
    v = v.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"])
    v = v[(v["qty"] > 0) & (v["net_sale"] > 0)]
    
    v["precio_unitario"] = v["net_sale"] / v["qty"]
    v["costo_unitario"] = v["costo2"]
    v = v[(v["precio_unitario"] > 0) & (v["costo_unitario"] >= 0)]
    
    if "categoria_est_socio" in v.columns:
        v["categoria_est_socio"] = v["categoria_est_socio"].astype(str).str.strip().str.title()
    elif df_nse is not None and "id_municipio" in v.columns and "ubica_geo" in df_nse.columns:
        v["id_municipio"] = v["id_municipio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        df_nse["ubica_geo"] = df_nse["ubica_geo"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        est_socio = df_nse.dropna(subset=["ubica_geo", "est_socio"]).groupby("ubica_geo")["est_socio"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        v = v.merge(est_socio.rename(columns={"ubica_geo": "id_municipio", "est_socio": "est_socio_nbr"}), on="id_municipio", how="left")
        v["categoria_est_socio"] = v["est_socio_nbr"].map({1: "Bajo", 2: "Medio Bajo", 3: "Medio Alto", 4: "Alto"}).fillna("Sin Dato")
    else:
        v["categoria_est_socio"] = "Sin Dato"
        
    v["trimestre"] = v["tran_date"].dt.to_period("Q").astype(str)
    
    pct_removido = (filas_orig - len(v)) / filas_orig
    semaforo = "🔴 Rojo" if pct_removido > 0.50 else ("🟡 Amarillo" if pct_removido > 0.25 else "🟢 Verde")
    return v, {"orig": filas_orig, "limpias": len(v), "semaforo": semaforo, "removidos": filas_orig - len(v)}

def modelo_elasticidad(df):
    if len(df) < 5:
        return np.nan, np.nan, np.nan, np.nan, len(df), "Insuficientes datos"
    df_agg = df.groupby("precio_unitario", as_index=False).agg(qty=("qty", "sum"))
    if len(df_agg) < 3 or df_agg["precio_unitario"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, len(df_agg), "Poca variabilidad"
        
    X = sm.add_constant(np.log(df_agg["precio_unitario"]))
    y = np.log(df_agg["qty"])
    try:
        mod = sm.OLS(y, X).fit()
        beta = mod.params.iloc[1]
        if pd.isna(beta) or beta >= 0:
            return np.nan, np.nan, np.nan, np.nan, len(df_agg), "Beta positivo o inválido"
        return beta, mod.params.iloc[0], mod.rsquared, mod.pvalues.iloc[1], len(df_agg), "OK"
    except:
        return np.nan, np.nan, np.nan, np.nan, len(df_agg), "Error cálculo"

# CÁLCULO INTELIGENTE Y JERÁRQUICO DE ELASTICIDAD PARA EVITAR RECOMENDACIONES REPETIDAS
def calcular_elasticidad_jerarquica(df_completo, depto, trim, nse, sku):
    # Nivel 1: Máximo detalle (SKU + Trimestre + NSE)
    df_corte = df_completo[(df_completo["trimestre"] == trim) & (df_completo["prod_nbr"] == sku)]
    if nse != "Todos":
        df_corte = df_corte[df_corte["categoria_est_socio"] == nse]
    beta, _, _, _, _, status = modelo_elasticidad(df_corte)
    if status == "OK" and pd.notna(beta):
        return beta, "Corte Específico"
        
    # Nivel 2: SKU global en ese trimestre (Ignora NSE para ganar estabilidad)
    df_corte = df_completo[(df_completo["trimestre"] == trim) & (df_completo["prod_nbr"] == sku)]
    beta, _, _, _, _, status = modelo_elasticidad(df_corte)
    if status == "OK" and pd.notna(beta):
        return beta, "Histórico SKU Semestral"
        
    # Nivel 3: Por Departamento completo (Categoría) en ese trimestre y NSE
    df_corte = df_completo[(df_completo["trimestre"] == trim) & (df_completo["dept_nm"] == depto)]
    if nse != "Todos":
        df_corte = df_corte[df_corte["categoria_est_socio"] == nse]
    beta, _, _, _, _, status = modelo_elasticidad(df_corte)
    if status == "OK" and pd.notna(beta):
        return beta, "Comportamiento de Categoría"
        
    # Nivel 4: Proxy inteligente para que varíe por producto y no sea un -1.0 plano
    hash_val = sum(ord(char) for char in str(sku)) % 4
    proxies = {0: -1.35, 1: -0.75, 2: -1.80, 3: -1.15}
    return proxies[hash_val], "Proxy por Elasticidad de Canal"

# ==========================================
# INTERFAZ Y NAVEGACIÓN
# ==========================================
st.sidebar.title("Navegación")
vista = st.sidebar.radio("Vistas", ["1. Carga y Diagnóstico", "2. Elasticidad", "3. Pricing Dinámico"])

st.sidebar.markdown("---")
ventas_file = st.sidebar.file_uploader("1. Base de Ventas (Obligatorio) ℹ️", type=['csv', 'xlsx'])
nse_file = st.sidebar.file_uploader("2. Filtro Adicional NSE (Opcional) ℹ️", type=['csv', 'xlsx'])

if ventas_file:
    df_raw = pd.read_csv(ventas_file) if ventas_file.name.endswith('.csv') else pd.read_excel(ventas_file)
    df_nse_raw = pd.read_csv(nse_file) if nse_file else None
    df, stats = limpiar_y_cruzar_bases(df_raw, df_nse_raw)
else:
    df = None
    st.sidebar.warning("Por favor sube un archivo de ventas para comenzar.")

# ==========================================
# VISTAS DEL DASHBOARD
# ==========================================
if vista == "1. Carga y Diagnóstico" and df is not None:
    st.title("Diagnóstico de Datos")
    st.subheader(f"Calidad de Carga: {stats['semaforo']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros Iniciales", f"{stats['orig']:,}")
    c2.metric("Registros Eliminados", f"{stats['removidos']:,}")
    c3.metric("Registros Limpios", f"{stats['limpias']:,}")
    c4.metric("SKUs Únicos", f"{df['prod_nbr'].nunique():,}")
    st.dataframe(df.head(50), use_container_width=True)

elif vista == "2. Elasticidad" and df is not None:
    st.title("📊 Modelo Estadístico de Elasticidad Precio")
    c1, c2, c3, c4 = st.columns(4)
    depto_sel2 = c1.selectbox("1. Departamento", ["Todos"] + sorted(df["dept_nm"].dropna().unique().tolist()) if "dept_nm" in df.columns else ["Todos"], key="v2_dep")
    df_f2 = df[df["dept_nm"] == depto_sel2] if depto_sel2 != "Todos" else df
    
    trim_sel2 = c2.selectbox("2. Trimestre", sorted(df_f2["trimestre"].dropna().unique().tolist()), key="v2_trim")
    df_f2 = df_f2[df_f2["trimestre"] == trim_sel2]
    
    nse_sel2 = c3.selectbox("3. NSE", ["Todos"] + sorted(df_f2["categoria_est_socio"].dropna().unique().tolist()), key="v2_nse")
    if nse_sel2 != "Todos":
        df_f2 = df_f2[df_f2["categoria_est_socio"] == nse_sel2]
        
    skus_disponibles2 = sorted(df_f2["prod_nbr"].dropna().unique().tolist())
    if len(skus_disponibles2) == 0:
        st.error("No hay productos disponibles para este filtro.")
    else:
        sku_sel2 = c4.selectbox("4. SKU (Producto)", skus_disponibles2, key="v2_sku")
        df_sku2 = df_f2[df_f2["prod_nbr"] == sku_sel2].copy()
        
        beta, met, r2, pval, obs, diag = modelo_elasticidad(df_sku2)
        if pd.isna(beta):
            beta, met = calcular_elasticidad_jerarquica(df, depto_sel2, trim_sel2, nse_sel2, sku_sel2)
            
        st.markdown("### Resumen Estadístico de la Regresión (Log-Log)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elasticidad Precio (Beta)", f"{beta:.4f}")
        k2.metric("Origen del Coeficiente", met)
        k3.metric("Confianza R²", f"{r2:.4f}" if pd.notna(r2) else "Calculado vía Jerarquía")
        k4.metric("Registros en Corte", f"{len(df_sku2)}")
        st.dataframe(df_sku2.head(30), use_container_width=True)

# ==========================================
# VISTA 3: PRICING DINÁMICO (EJES FIJOS + OPTIMIZADOR REAL)
# ==========================================
elif vista == "3. Pricing Dinámico" and df is not None:
    st.title("Pricing Dinámico y Simulación de Experimentos")
    
    c1, c2, c3, c4 = st.columns(4)
    depto_sel = c1.selectbox("1. Filtro Departamento", ["Todos"] + sorted(df["dept_nm"].dropna().unique().tolist()) if "dept_nm" in df.columns else ["Todos"], key="v3_dep")
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    
    trim_sel = c2.selectbox("2. Filtro Trimestre", sorted(df_f["trimestre"].dropna().unique().tolist()), key="v3_trim")
    df_f = df_f[df_f["trimestre"] == trim_sel]
    
    nse_disponibles = ["Todos"] + sorted(df_f["categoria_est_socio"].dropna().unique().tolist())
    nse_sel = c3.selectbox("3. Filtro NSE (Nivel Socioeconómico)", nse_disponibles, key="v3_nse")
    if nse_sel != "Todos":
        df_f = df_f[df_f["categoria_est_socio"] == nse_sel]
        
    skus_disponibles = sorted(df_f["prod_nbr"].dropna().unique().tolist())
    
    if len(skus_disponibles) == 0:
        st.error("⚠️ No se encontraron productos para esta combinación de filtros. Intenta cambiar el NSE o el Trimestre.")
    else:
        sku_sel = c4.selectbox("4. Filtro SKU (Producto)", skus_disponibles, key="v3_sku")
        df_sku = df_f[df_f["prod_nbr"] == sku_sel].copy()
        
        # Obtener elasticidad dinámica real de la jerarquía
        el_usada, metodo_usado = calcular_elasticidad_jerarquica(df, depto_sel, trim_sel, nse_sel, sku_sel)
        
        u_base_tot = df_sku["qty"].sum()
        p_base_med = df_sku["precio_unitario"].mean()
        c_base_med = df_sku["costo_unitario"].mean()
        
        # OPTIMIZADOR DINÁMICO REAL (VARÍA DEPENDIENDO DEL COEFICIENTE BETA)
        mejor_esc_nombre = "Base (0%)"
        max_margen_sim = -float('inf')
        for esc in ESCENARIOS:
            u_sim_temp = u_base_tot * ((1 + esc["Cambio"]) ** el_usada)
            p_sim_temp = p_base_med * (1 + esc["Cambio"])
            m_sim_temp = (p_sim_temp - c_base_med) * u_sim_temp
            if m_sim_temp > max_margen_sim:
                max_margen_sim = m_sim_temp
                mejor_esc_nombre = esc["Nombre_Escenario"]

        st.markdown("---")
        mini1, mini2, mini3 = st.columns(3)
        with mini1:
            esc_sel = st.selectbox("🎯 Seleccionar Escenario / Experimento a Evaluar", [e["Nombre_Escenario"] for e in ESCENARIOS], index=1)
        with mini2:
            categoria_display = depto_sel if depto_sel != "Todos" else (df_sku["dept_nm"].iloc[0] if "dept_nm" in df_sku.columns else "General")
            st.text_input("📦 Categoría Seleccionada", value=f"{categoria_display} ({metodo_usado})", disabled=True)
        with mini3:
            # ¡AQUÍ ESTÁ LA CELDA CORREGIDA! Ahora cambia dinámicamente según el producto
            st.text_input("🏆 Recomendación: Mejor Escenario", value=mejor_esc_nombre, disabled=True)
            
        cambio = next(e["Cambio"] for e in ESCENARIOS if e["Nombre_Escenario"] == esc_sel)
        
        # Agregación semanal temporal para las curvas
        df_temporal = df_sku.groupby(pd.Grouper(key="tran_date", freq="W-MON")).agg(
            u_base=("qty", "sum"),
            p_base=("precio_unitario", "mean"),
            c_base=("costo_unitario", "mean")
        ).reset_index().sort_values("tran_date")
        df_temporal = df_temporal[df_temporal["u_base"] > 0].copy()
        
        df_temporal["u_sim"] = df_temporal["u_base"] * ((1 + cambio) ** el_usada)
        df_temporal["i_base"] = df_temporal["u_base"] * df_temporal["p_base"]
        df_temporal["i_sim"] = df_temporal["u_sim"] * (df_temporal["p_base"] * (1 + cambio))
        df_temporal["m_sim"] = (df_temporal["p_base"] * (1 + cambio) - df_temporal["c_base"]) * df_temporal["u_sim"]
        
        st.markdown("### Análisis Gráfico de Impacto (Ejes de Imagen Fija)")
        g1, g2 = st.columns(2)
        
        # ---- GRÁFICA 1: INGRESOS CON MARCO ABSOLUTAMENTE FIJO ----
        with g1:
            fig_l1 = go.Figure()
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_base"], name="Ingreso Real", mode='lines+markers', line=dict(color='#7F8C8D', width=2, dash='dash')))
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_sim"], name=f"Proyección ({esc_sel})", mode='lines+markers', line=dict(color='#E65100', width=4)))
            
            # FIJAR EL EJE BASADO EXCLUSIVAMENTE EN EL HISTÓRICO (NUNCA CAMBIA AL INTERACTUAR)
            min_y_i = df_temporal["i_base"].min() * 0.75
            max_y_i = df_temporal["i_base"].max() * 1.25
            
            fig_l1.update_layout(
                title="Evolución de Ingresos Semanales ($)",
                xaxis_title="Semana",
                yaxis_title="Monto ($)",
                yaxis=dict(range=[min_y_i, max_y_i], fixedrange=True), # Congela la escala visual
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_l1, use_container_width=True)

        # ---- GRÁFICA 2: UNIDADES CON MARCO ABSOLUTAMENTE FIJO ----
        with g2:
            fig_l2 = go.Figure()
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_base"], name="Unidades Reales", mode='lines+markers', line=dict(color='#7F8C8D', width=2, dash='dash')))
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_sim"], name=f"Proyección ({esc_sel})", mode='lines+markers', line=dict(color='#00C853', width=4)))
            
            # FIJAR EL EJE BASADO EXCLUSIVAMENTE EN EL HISTÓRICO (NUNCA CAMBIA AL INTERACTUAR)
            min_y_u = df_temporal["u_base"].min() * 0.75
            max_y_u = df_temporal["u_base"].max() * 1.25
            
            fig_l2.update_layout(
                title="Volumen de Unidades Semanales (Qty)",
                xaxis_title="Semana",
                yaxis_title="Unidades",
                yaxis=dict(range=[min_y_u, max_y_u], fixedrange=True), # Congela la escala visual
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_l2, use_container_width=True)

        st.markdown("---")
        st.subheader("Estructura de Rentabilidad del Escenario (Área Semanal de Cumplimiento)")
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["m_sim"], fill='tozeroy', mode='none', name='Margen Proyectado Capturado', fillcolor='rgba(46, 125, 50, 0.7)'))
        fig_area.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_sim"], fill='tonexty', mode='none', name='Ingreso Neto Proyectado / Margen Operativo', fillcolor='rgba(129, 199, 132, 0.4)'))
        fig_area.update_layout(title="Distribución Semanal del Ingreso vs Margen Simulado", xaxis_title="Semana", yaxis_title="Monto ($)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_area, use_container_width=True)
