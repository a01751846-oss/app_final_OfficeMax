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
    
    # Conversión de fechas y numéricos
    v["tran_date"] = pd.to_datetime(v["tran_date"], errors="coerce")
    for col in ["qty", "net_sale", "costo2"]:
        if col in v.columns:
            v[col] = pd.to_numeric(v[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        
    v = v.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"])
    v = v[(v["qty"] > 0) & (v["net_sale"] > 0)]
    
    v["precio_unitario"] = v["net_sale"] / v["qty"]
    v["costo_unitario"] = v["costo2"]
    v = v[(v["precio_unitario"] > 0) & (v["costo_unitario"] >= 0)]
    
    # INTEGRACIÓN NATIVA DEL NSE DESDE TU BASE
    if "categoria_est_socio" in v.columns:
        # Convierte "medio alto" a "Medio Alto" para estandarizar los filtros
        v["categoria_est_socio"] = v["categoria_est_socio"].astype(str).str.strip().str.title()
    elif df_nse is not None and "id_municipio" in v.columns and "ubica_geo" in df_nse.columns:
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
        return -1.0, np.nan, np.nan, np.nan, len(df_agg), "No suficiente variabilidad (Usando Base -1.0)"
        
    X = sm.add_constant(np.log(df_agg["precio_unitario"]))
    y = np.log(df_agg["qty"])
    
    try:
        mod = sm.OLS(y, X).fit()
        beta = mod.params.iloc[1]
        if pd.isna(beta) or beta >= 0:
            beta = -1.0
        return beta, mod.params.iloc[0], mod.rsquared, mod.pvalues.iloc[1], len(df_agg), "OK"
    except:
        return -1.0, np.nan, np.nan, np.nan, len(df_agg), "Error (Usando Base -1.0)"

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
# VISTA 1 Y VISTA 2
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
    st.title("Modelo de Elasticidad Precio de la Demanda")
    st.info("Configura los filtros en la Vista 3 para ejecutar simulaciones completas.")

# ==========================================
# VISTA 3: PRICING DINÁMICO (FILTROS EN CASCADA CORREGIDOS)
# ==========================================
elif vista == "3. Pricing Dinámico" and df is not None:
    st.title("Pricing Dinámico y Simulación de Experimentos")
    
    # FILTROS EN CASCADA SUPERIORES
    c1, c2, c3, c4 = st.columns(4)
    
    # 1. Filtro Departamento
    deptos_disponibles = ["Todos"] + sorted(df["dept_nm"].dropna().unique().tolist()) if "dept_nm" in df.columns else ["Todos"]
    depto_sel = c1.selectbox("1. Filtro Departamento", deptos_disponibles)
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    
    # 2. Filtro Trimestre (Depende del departamento elegido)
    trim_disponibles = sorted(df_f["trimestre"].dropna().unique().tolist())
    trim_sel = c2.selectbox("2. Filtro Trimestre", trim_disponibles)
    df_f = df_f[df_f["trimestre"] == trim_sel]
    
    # 3. Filtro NSE (Muestra dinámicamente los NSE reales de tu archivo)
    nse_disponibles = ["Todos"] + sorted(df_f["categoria_est_socio"].dropna().unique().tolist())
    nse_sel = c3.selectbox("3. Filtro NSE (Nivel Socioeconómico)", nse_disponibles)
    if nse_sel != "Todos":
        df_f = df_f[df_f["categoria_est_socio"] == nse_sel]
        
    # 4. Filtro SKU (Solo muestra SKUs existentes para los cortes de arriba)
    skus_disponibles = sorted(df_f["prod_nbr"].dropna().unique().tolist())
    
    if len(skus_disponibles) == 0:
        st.error("⚠️ No se encontraron productos para esta combinación de filtros. Intenta cambiar el NSE o el Trimestre.")
    else:
        sku_sel = c4.selectbox("4. Filtro SKU (Producto)", skus_disponibles)
        
        df_sku = df_f[df_f["prod_nbr"] == sku_sel].copy()
        
        # Calcular Elasticidad real para este grupo
        beta, _, _, _, _, _ = modelo_elasticidad(df_sku)
        el_usada = np.clip(beta, -5, -0.1) if pd.notna(beta) else -1.0
        
        # Totales Históricos Base
        u_base_tot = df_sku["qty"].sum()
        p_base_med = df_sku["precio_unitario"].mean()
        c_base_med = df_sku["costo_unitario"].mean()
        
        # Optimización: Encontrar el escenario con mayor margen simulado
        mejor_esc_nombre = "Mantener precio"
        max_margen_sim = -float('inf')
        
        for esc in ESCENARIOS:
            u_sim_temp = u_base_tot * ((1 + esc["Cambio"]) ** el_usada)
            p_sim_temp = p_base_med * (1 + esc["Cambio"])
            m_sim_temp = (p_sim_temp - c_base_med) * u_sim_temp
            if m_sim_temp > max_margen_sim:
                max_margen_sim = m_sim_temp
                mejor_esc_nombre = esc["Nombre_Escenario"]

        # ==========================================
        # LAS 3 MINI CASILLAS ALINEADAS (REQUISITO)
        # ==========================================
        st.markdown("---")
        mini1, mini2, mini3 = st.columns(3)
        with mini1:
            esc_sel = st.selectbox("🎯 Seleccionar Escenario / Experimento a Evaluar", [e["Nombre_Escenario"] for e in ESCENARIOS], index=1)
        with mini2:
            categoria_display = depto_sel if depto_sel != "Todos" else (df_sku["dept_nm"].iloc[0] if "dept_nm" in df_sku.columns else "General")
            st.text_input("📦 Categoría Seleccionada", value=categoria_display, disabled=True)
        with mini3:
            st.text_input("🏆 Recomendación: Mejor Escenario", value=mejor_esc_nombre, disabled=True)
            
        cambio = next(e["Cambio"] for e in ESCENARIOS if e["Nombre_Escenario"] == esc_sel)
        
        # ==========================================
        # AGREGACIÓN SEMANAL Y SIMULACIÓN DE ESCENARIOS
        # ==========================================
        df_temporal = df_sku.groupby(pd.Grouper(key="tran_date", freq="W-MON")).agg(
            u_base=("qty", "sum"),
            p_base=("precio_unitario", "mean"),
            c_base=("costo_unitario", "mean")
        ).reset_index().sort_values("tran_date")
        
        df_temporal = df_temporal[df_temporal["u_base"] > 0].copy()
        
        # Aplicación estricta de la proyección matemática basada en el cambio seleccionado
        df_temporal["u_sim"] = df_temporal["u_base"] * ((1 + cambio) ** el_usada)
        df_temporal["i_base"] = df_temporal["u_base"] * df_temporal["p_base"]
        df_temporal["i_sim"] = df_temporal["u_sim"] * (df_temporal["p_base"] * (1 + cambio))
        df_temporal["m_sim"] = (df_temporal["p_base"] * (1 + cambio) - df_temporal["c_base"]) * df_temporal["u_sim"]
        
        # Métricas consolidadas finales
        tot_i_real = df_temporal["i_base"].sum()
        tot_i_sim = df_temporal["i_sim"].sum()
        tot_u_real = df_temporal["u_base"].sum()
        tot_u_sim = df_temporal["u_sim"].sum()
        tot_m_sim = df_temporal["m_sim"].sum()

        st.markdown("### Análisis Gráfico de Impacto")
        g1, g2 = st.columns(2)
        
        # ---- GRÁFICA 1: LÍNEAS DE INGRESO ----
        with g1:
            fig_l1 = go.Figure()
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_base"], name="Ingreso Real", mode='lines+markers', line=dict(color='#4A4A4A', width=2, dash='dash')))
            fig_l1.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["i_sim"], name=f"Proyección ({esc_sel})", mode='lines+markers', line=dict(color='#E65100', width=4)))
            fig_l1.update_layout(title="Evolución de Ingresos Semanales ($)", xaxis_title="Semana", yaxis_title="Monto ($)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_l1, use_container_width=True)
            
            st.write(f"📝 **Explicación:** Esta gráfica compara el **Ingreso Real** histórico contra el **Ingreso Proyectado** que ocurriría aplicando el experimento *{esc_sel}*. Con base en los datos de la app, el ingreso real acumulado en este periodo fue de **${tot_i_real:,.2f}**, mientras que la simulación proyecta un total de **${tot_i_sim:,.2f}**.")

        # ---- GRÁFICA 2: LÍNEAS DE UNIDADES ----
        with g2:
            fig_l2 = go.Figure()
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_base"], name="Unidades Reales", mode='lines+markers', line=dict(color='#4A4A4A', width=2, dash='dash')))
            fig_l2.add_trace(go.Scatter(x=df_temporal["tran_date"], y=df_temporal["u_sim"], name=f"Proyección ({esc_sel})", mode='lines+markers', line=dict(color='#00C853', width=4)))
            fig_l2.update_layout(title="Volumen de Unidades Semanales (Qty)", xaxis_title="Semana", yaxis_title="Unidades", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_l2, use_container_width=True)
            
            st.write(f"📝 **Explicación:** Compara el volumen físico de ventas semanales reales vs las **Unidades Proyectadas** considerando el impacto de la elasticidad estimada en **{el_usada:.2f}**. El histórico acumuló **{tot_u_real:,.0f}** piezas, y el experimento seleccionado proyecta un volumen de **{tot_u_sim:,.0f}** piezas.")

        st.markdown("---")
        
        # ---- GRÁFICA 3: GRÁFICO DE ÁREA DE RENTABILIDAD ----
        st.subheader("Estructura de Rentabilidad del Escenario (Área Semanal de Cumplimiento)")
        
        fig_area = go.Figure()
        
        # Capa 1: Margen Proyectado Puro (Verde Oscuro / Rentabilidad Capturada que sí se alcanzó)
        fig_area.add_trace(go.Scatter(
            x=df_temporal["tran_date"], y=df_temporal["m_sim"],
            fill='tozeroy', mode='none', name='Margen Proyectado Capturado',
            fillcolor='rgba(46, 125, 50, 0.7)' 
        ))
        
        # Capa 2: Ingreso Neto Total Proyectado (Verde Claro / Brecha operativa o excedente sobre el costo)
        fig_area.add_trace(go.Scatter(
            x=df_temporal["tran_date"], y=df_temporal["i_sim"],
            fill='tonexty', mode='none', name='Ingreso Neto Proyectado / Margen Operativo',
            fillcolor='rgba(129, 199, 132, 0.4)' 
        ))
        
        fig_area.update_layout(
            title="Distribución Semanal del Ingreso vs Margen Simulado",
            xaxis_title="Semana", yaxis_title="Monto ($)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_area, use_container_width=True)
        
        st.write(f"ℹ️ **Interpretación del Gráfico de Área:** Esta visualización continua detalla la proporción del margen simulado dentro del volumen total de ingresos. El área inferior (**Verde Oscuro**) representa la utilidad neta pura proyectada que captura el negocio (**${tot_m_sim:,.2f}** totales), mientras que la franja superior (**Verde Claro**) representa el ingreso total bruto simulado necesario para sostener la operación del SKU **{sku_sel}** bajo las condiciones de este nivel socioeconómico.")
