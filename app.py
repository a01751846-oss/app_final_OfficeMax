import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="App Pricing & Elasticidad", layout="wide")
COLOR_PALETTE = ['#1f77b4', '#9467bd', '#00b1b2', '#2ca02c', '#17becf']

ESCENARIOS = [
    {"Nombre_Escenario": "Cambio de precio +15%", "Cambio": 0.15, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Cambio de precio +10%", "Cambio": 0.10, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Cambio de precio +5%", "Cambio": 0.05, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Base (0%)", "Cambio": 0.00, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Cambio de precio -5%", "Cambio": -0.05, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Cambio de precio -10%", "Cambio": -0.10, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Cambio de precio -15%", "Cambio": -0.15, "Tipo": "Cambio Precio"},
    {"Nombre_Escenario": "Promoción 2do al 50%", "Cambio": -0.25, "Tipo": "Promoción"},
    {"Nombre_Escenario": "Promoción 3x2", "Cambio": -0.3333, "Tipo": "Promoción"},
    {"Nombre_Escenario": "Promoción 2x1", "Cambio": -0.50, "Tipo": "Promoción"},
]

# ==========================================
# FUNCIONES MATEMÁTICAS Y DE NEGOCIO
# ==========================================
@st.cache_data
def limpiar_y_cruzar_bases(df_ventas, df_nse):
    v = df_ventas.copy()
    v.columns = v.columns.str.strip()
    filas_orig = len(v)
    
    v["tran_date"] = pd.to_datetime(v["tran_date"], errors="coerce")
    for col in ["qty", "net_sale", "costo2"]:
        v[col] = pd.to_numeric(v[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        
    v = v.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"])
    v = v[(v["qty"] > 0) & (v["net_sale"] > 0)]
    
    v["precio_unitario"] = v["net_sale"] / v["qty"]
    v["costo_unitario"] = v["costo2"]
    v = v[(v["precio_unitario"] > 0) & (v["costo_unitario"] >= 0)]
    
    if df_nse is not None and "id_municipio" in v.columns and "ubica_geo" in df_nse.columns:
        v["id_municipio"] = v["id_municipio"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        df_nse["ubica_geo"] = df_nse["ubica_geo"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        est_socio = df_nse.dropna(subset=["ubica_geo", "est_socio"]).groupby("ubica_geo")["est_socio"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        v = v.merge(est_socio.rename(columns={"ubica_geo": "id_municipio", "est_socio": "est_socio_nbr"}), on="id_municipio", how="left")
        v["categoria_est_socio"] = v["est_socio_nbr"].map({1: "Bajo", 2: "Medio Bajo", 3: "Medio Alto", 4: "Alto", "1": "Bajo", "2": "Medio Bajo", "3": "Medio Alto", "4": "Alto"}).fillna("Sin Dato")
    else:
        v["categoria_est_socio"] = "Sin Dato"
        
    v["trimestre"] = v["tran_date"].dt.to_period("Q").astype(str)
    
    pct_removido = (filas_orig - len(v)) / filas_orig
    semaforo = "🔴 Rojo" if pct_removido > 0.50 else ("🟡 Amarillo" if pct_removido > 0.25 else "🟢 Verde")
    return v, {"orig": filas_orig, "limpias": len(v), "semaforo": semaforo, "removidos": filas_orig - len(v)}

def modelo_elasticidad(df):
    df_agg = df.groupby("precio_unitario", as_index=False).agg(qty=("qty", "sum"))
    if len(df_agg) < 3 or df_agg["precio_unitario"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, len(df_agg), "No evaluable"
        
    X = sm.add_constant(np.log(df_agg["precio_unitario"]))
    y = np.log(df_agg["qty"])
    
    try:
        mod = sm.OLS(y, X).fit()
        return mod.params.iloc[1], mod.params.iloc[0], mod.rsquared, mod.pvalues.iloc[1], len(df_agg), "OK"
    except:
        return np.nan, np.nan, np.nan, np.nan, len(df_agg), "Error Modelo"

# ==========================================
# INTERFAZ Y SIDEBAR
# ==========================================
st.sidebar.title("Navegación")
vista = st.sidebar.radio("Vistas", ["1. Carga y Diagnóstico", "2. Elasticidad", "3. Pricing Dinámico"])

st.sidebar.markdown("---")
ventas_file = st.sidebar.file_uploader("1. Ventas (Obligatorio) ℹ️", type=['csv', 'xlsx'])
nse_file = st.sidebar.file_uploader("2. Nivel Socioeconómico (Opcional) ℹ️", type=['csv', 'xlsx'])

if ventas_file:
    df_raw = pd.read_csv(ventas_file) if ventas_file.name.endswith('.csv') else pd.read_excel(ventas_file)
    df_nse_raw = pd.read_csv(nse_file) if nse_file else None
    df, stats = limpiar_y_cruzar_bases(df_raw, df_nse_raw)
else:
    df = None
    st.sidebar.warning("Sube el archivo de ventas para habilitar la app.")

# ==========================================
# VISTA 1: CARGA
# ==========================================
if vista == "1. Carga y Diagnóstico" and df is not None:
    st.title("Diagnóstico de Datos")
    st.subheader(f"Calidad: {stats['semaforo']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros Iniciales", stats["orig"])
    c2.metric("Registros Eliminados", stats["removidos"])
    c3.metric("Registros Limpios", stats["limpias"])
    c4.metric("SKUs Únicos", df["prod_nbr"].nunique())
    st.dataframe(df.head(50), use_container_width=True)

# ==========================================
# VISTA 2: ELASTICIDAD
# ==========================================
elif vista == "2. Elasticidad" and df is not None:
    st.title("Modelo de Elasticidad (Log-Log)")
    c1, c2, c3 = st.columns(3)
    deptos = df["dept_nm"].dropna().unique().tolist() if "dept_nm" in df.columns else ["N/A"]
    depto_sel = c1.selectbox("Departamento", ["Todos"] + deptos)
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    trimestres = df_f["trimestre"].dropna().unique().tolist()
    trim_sel = c2.selectbox("Trimestre", ["Todos"] + trimestres)
    df_f2 = df_f[df_f["trimestre"] == trim_sel] if trim_sel != "Todos" else df_f
    skus = df_f2["prod_nbr"].dropna().unique().tolist()
    sku_sel = c3.selectbox("SKU", skus)
    
    if sku_sel:
        df_sku = df_f2[df_f2["prod_nbr"] == sku_sel]
        beta, alfa, r2, pval, obs, diag = modelo_elasticidad(df_sku)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elasticidad (Beta)", f"{beta:.3f}" if pd.notna(beta) else "N/A")
        k2.metric("R² del Modelo", f"{r2:.2f}" if pd.notna(r2) else "N/A")
        st.dataframe(df_sku.head(10))

# ==========================================
# VISTA 3: PRICING DINÁMICO (AGRUPACIÓN SEMANAL)
# ==========================================
elif vista == "3. Pricing Dinámico" and df is not None:
    st.title("Pricing Dinámico y Simulación")
    
    # Filtros principales superiores
    c1, c2, c3, c4 = st.columns(4)
    depto_sel = c1.selectbox("Filtro Departamento", ["Todos"] + df["dept_nm"].dropna().unique().tolist() if "dept_nm" in df.columns else ["Todos"])
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    trim_sel = c2.selectbox("Filtro Trimestre", df_f["trimestre"].dropna().unique().tolist())
    nse_sel = c3.selectbox("Filtro NSE", ["Todos"] + df_f["categoria_est_socio"].dropna().unique().tolist())
    
    df_f = df_f[df_f["trimestre"] == trim_sel]
    if nse_sel != "Todos": df_f = df_f[df_f["categoria_est_socio"] == nse_sel]
    
    sku_sel = c4.selectbox("Filtro SKU (Unidad)", df_f["prod_nbr"].dropna().unique().tolist())
    
    st.markdown("---")
    if sku_sel:
        df_sku = df_f[df_f["prod_nbr"] == sku_sel].copy()
        beta, _, _, _, _, _ = modelo_elasticidad(df_sku)
        el_usada = np.clip(beta, -5, 0) if pd.notna(beta) and beta < 0 else -1.0
        
        # Pre-cálculo para optimización del mejor escenario
        u_base_tot = df_sku["qty"].sum()
        p_base_med = df_sku["precio_unitario"].mean()
        c_base_med = df_sku["costo_unitario"].mean()
        
        mejor_esc_nombre = "Mantener precio"
        max_margen_sim = -float('inf')
        
        for esc in ESCENARIOS:
            u_sim_temp = u_base_tot * np.exp(el_usada * np.log1p(esc["Cambio"]))
            p_sim_temp = p_base_med * (1 + esc["Cambio"])
            m_sim_temp = (p_sim_temp - c_base_med) * u_sim_temp
            if m_sim_temp > max_margen_sim:
                max_margen_sim = m_sim_temp
                mejor_esc_nombre = esc["Nombre_Escenario"]

        # ==========================================
        # LAS 3 MINI CASILLAS ALINEADAS
        # ==========================================
        mini1, mini2, mini3 = st.columns(3)
        with mini1:
            esc_sel = st.selectbox("🎯 Seleccionar Escenario", [e["Nombre_Escenario"] for e in ESCENARIOS])
        with mini2:
            categoria_display = depto_sel if depto_sel != "Todos" else (df_sku["dept_nm"].iloc[0] if "dept_nm" in df_sku.columns else "General")
            st.text_input("📦 Categoría del Producto", value=categoria_display, disabled=True)
        with mini3:
            st.text_input("🏆 Mejor Escenario (Qué hacer)", value=mejor_esc_nombre, disabled=True)
            
        cambio = next(e["Cambio"] for e in ESCENARIOS if e["Nombre_Escenario"] == esc_sel)
        
        # ==========================================
        # AGRUPACIÓN SEMANAL PARA LAS GRÁFICAS
        # ==========================================
        # Se agrupa por semana (W-MON) para evitar gráficos vacíos o puntos desconectados
        df_temporal = df_sku.groupby(pd.Grouper(key="tran_date", freq="W-MON")).agg(
            u_base=("qty", "sum"),
            p_base=("precio_unitario", "mean"),
            c_base=("costo_unitario", "mean")
        ).reset_index().sort_values("tran_date")
        
        # Filtrar semanas vacías donde no hubo ventas
        df_temporal = df_temporal[df_temporal["u_base"] > 0].copy()
        
        # Cálculos semanales
        df_temporal["i_base"] = df_temporal["u_base"] * df_temporal["p_base"]
        df_temporal["u_sim"] = df_temporal["u_base"] * np.exp(el_usada * np.log1p(cambio))
        df_temporal["i_sim"] = df_temporal["u_sim"] * (df_temporal["p_base"] * (1 + cambio))
        df_temporal["m_sim"] = (df_temporal["p_base"] * (1 + cambio) - df_temporal["c_base"]) * df_temporal["u_sim"]
        
        # Totales para los textos
        tot_i_real = df_temporal["i_base"].sum()
        tot_i_sim = df_temporal["i_sim"].sum()
        tot_u_real = df_temporal["u_base"].sum()
        tot_u_sim = df_temporal["u_sim"].sum()
        tot_m_sim = df_temporal["m_sim"].sum()

        st.markdown("### Análisis Gráfico Semanal")
        g1, g2 = st.columns(2)
        
        # ---- GRÁFICA 1: LÍNEAS DE INGRESO ----
        with g1:
            fig_l1 = go.Figure()
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_base"], name="Ingreso Real", mode='lines+markers', line=dict(color='gray', width=2)))
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_sim"], name="Ingreso Proyectado", mode='lines+markers', line=dict(color='#1f77b4', width=3)))
            fig_l1.update_layout(title="Ingresos Semanales ($)", xaxis_title="Semana", yaxis_title="Monto ($)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_l1, use_container_width=True)
            
            st.write(f"📝 **Explicación:** Esta gráfica compara los **Ingresos Reales** históricos contra los **Ingresos Proyectados** bajo el escenario *{esc_sel}*. El ingreso histórico total fue de **${tot_i_real:,.2f}**, mientras que el modelo estima un ingreso de **${tot_i_sim:,.2f}**.")

        # ---- GRÁFICA 2: LÍNEAS DE UNIDADES ----
        with g2:
            fig_l2 = go.Figure()
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_base"], name="Unidades Reales", mode='lines+markers', line=dict(color='gray', width=2)))
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_sim"], name="Unidades Proyectadas", mode='lines+markers', line=dict(color='#9467bd', width=3)))
            fig_l2.update_layout(title="Unidades Semanales (Qty)", xaxis_title="Semana", yaxis_title="Unidades", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_l2, use_container_width=True)
            
            st.write(f"📝 **Explicación:** Mide el volumen de ventas por semana. Compara las **Unidades Reales** frente a las **Unidades Proyectadas** considerando una elasticidad de **{el_usada:.2f}**. El volumen real fue de **{tot_u_real:,.0f}** piezas vs **{tot_u_sim:,.0f}** piezas proyectadas.")

        st.markdown("---")
        
        # ---- GRÁFICA 3: ÁREA INGRESO VS MARGEN SIMULADO ----
        st.subheader("Estructura de Rentabilidad del Escenario (Área Semanal)")
        
        fig_area = go.Figure()
        
        # Área de Margen (Turquesa oscuro / Verde oscuro)
        fig_area.add_trace(go.Scatter(
            x=df_temporal["tran_date"], y=df_temporal["m_sim"],
            fill='tozeroy', mode='none', name='Margen Proyectado Alcanzado',
            fillcolor='rgba(0, 177, 178, 0.6)' 
        ))
        
        # Área de Ingreso Total (Verde claro)
        fig_area.add_trace(go.Scatter(
            x=df_temporal["tran_date"], y=df_temporal["i_sim"],
            fill='tonexty', mode='none', name='Ingreso Total / Brecha Operativa',
            fillcolor='rgba(44, 160, 44, 0.3)' 
        ))
        
        fig_area.update_layout(
            title="Distribución del Ingreso vs Margen Simulado",
            xaxis_title="Semana", yaxis_title="Monto ($)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_area, use_container_width=True)
        
        st.write(f"ℹ️ **Interpretación:** El área inferior (**Turquesa**) representa la rentabilidad pura (margen) capturada por semana (**${tot_m_sim:,.2f}** totales), mientras que el área superior (**Verde Claro**) muestra el volumen total de ingresos proyectados. La diferencia entre ambas curvas representa el costo operativo o costo de mercancía.")

        # Botones de descarga
        st.markdown("---")
        experimentos = []
        for esc in ESCENARIOS:
            u = u_base_tot * np.exp(el_usada * np.log1p(esc["Cambio"]))
            i = (p_base_med * (1 + esc["Cambio"])) * u
            m = ((p_base_med * (1 + esc["Cambio"])) - c_base_med) * u
            experimentos.append({
                'SKU': sku_sel, 'dept_nm': categoria_display, 'trimestre': trim_sel, 'escenario aplicado': esc["Nombre_Escenario"],
                'unidades simuladas': u, 'ingreso simulado': i, 'margen simulado': m
            })
        df_exp = pd.DataFrame(experimentos)
        df_exp['mejor escenario'] = mejor_esc_nombre
        
        d1, d2 = st.columns(2)
        d1.download_button("📥 Descargar Todos los Experimentos", df_exp.to_csv(index=False).encode('utf-8'), "todos_experimentos.csv", "text/csv")
        d2.download_button("🏆 Descargar Solo el Mejor Escenario", df_exp[df_exp["escenario aplicado"] == mejor_esc_nombre].to_csv(index=False).encode('utf-8'), "mejor_escenario.csv", "text/csv")
