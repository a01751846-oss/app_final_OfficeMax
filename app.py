import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import re
import unicodedata

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
    
    # Limpieza Básica
    v["tran_date"] = pd.to_datetime(v["tran_date"], errors="coerce")
    for col in ["qty", "net_sale", "costo2"]:
        v[col] = pd.to_numeric(v[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
        
    v = v.dropna(subset=["tran_date", "qty", "net_sale", "prod_nbr", "costo2"])
    v = v[(v["qty"] > 0) & (v["net_sale"] > 0)]
    
    v["precio_unitario"] = v["net_sale"] / v["qty"]
    v["costo_unitario"] = v["costo2"]
    v = v[(v["precio_unitario"] > 0) & (v["costo_unitario"] >= 0)]
    
    # Cruce NSE 
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
    """Regresión Log-Log usando Statsmodels."""
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
ventas_file = st.sidebar.file_uploader("1. Ventas (Obligatorio) ℹ️", type=['csv', 'xlsx'], help="Base con tran_date, qty, net_sale, prod_nbr.")
nse_file = st.sidebar.file_uploader("2. Nivel Socioeconómico (Opcional) ℹ️", type=['csv', 'xlsx'], help="Base de hogares INEGI.")

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
    st.info("ℹ️ Explora la sensibilidad de la demanda ante cambios en el precio.")
    
    c1, c2, c3 = st.columns(3)
    deptos = df["dept_nm"].dropna().unique().tolist() if "dept_nm" in df.columns else ["N/A"]
    depto_sel = c1.selectbox("Departamento ℹ️", ["Todos"] + deptos, help="Filtra categoría principal.")
    
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    trimestres = df_f["trimestre"].dropna().unique().tolist()
    trim_sel = c2.selectbox("Trimestre", ["Todos"] + trimestres)
    
    df_f2 = df_f[df_f["trimestre"] == trim_sel] if trim_sel != "Todos" else df_f
    skus = df_f2["prod_nbr"].dropna().unique().tolist()
    sku_sel = c3.selectbox("SKU", skus)
    
    if sku_sel:
        df_sku = df_f2[df_f2["prod_nbr"] == sku_sel]
        beta, alfa, r2, pval, obs, diag = modelo_elasticidad(df_sku)
        
        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elasticidad (Beta)", f"{beta:.3f}" if pd.notna(beta) else "N/A", "Elástica" if beta < -1 else "Inelástica")
        k2.metric("R² del Modelo", f"{r2:.2f}" if pd.notna(r2) else "N/A")
        k3.metric("Observaciones", obs)
        k4.metric("Diagnóstico", diag)
        
        g1, g2, g3 = st.columns(3)
        with g1:
            fig1 = px.scatter(df_sku, x="qty", y="precio_unitario", trendline="ols", title="Curva de Elasticidad", color_discrete_sequence=[COLOR_PALETTE[0]])
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Relación Precio vs Cantidad (OLS)")
        with g2:
            fig2 = px.line(df_sku.groupby("tran_date")["qty"].sum().reset_index(), x="tran_date", y="qty", title="Demanda Histórica", color_discrete_sequence=[COLOR_PALETTE[1]])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Unidades vendidas a través del tiempo.")
        with g3:
            if "estado" in df_sku.columns:
                fig3 = px.bar(df_sku.groupby("estado")["qty"].sum().reset_index(), x="estado", y="qty", title="Demanda por Estado", color_discrete_sequence=[COLOR_PALETTE[2]])
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Concentración geográfica del SKU.")
                
        # Descarga CSV Vista 2
        df_desc = pd.DataFrame([{
            'SKU': sku_sel, 'dept_nm': depto_sel, 'subdept_nm': df_sku['subdept_nm'].iloc[0] if 'subdept_nm' in df_sku else '',
            'marca': df_sku['marca'].iloc[0] if 'marca' in df_sku else '', 'tipo_marca': df_sku['tipo_marca'].iloc[0] if 'tipo_marca' in df_sku else '',
            'categoria_est_socio': df_sku['categoria_est_socio'].iloc[0], 'trimestre': trim_sel,
            'beta': beta, 'elasticidad': beta, 'alfa': alfa, 'r2': r2, 'p-value': pval, 'observaciones': obs, 'diagnostico': diag
        }])
        st.download_button("📥 Descargar Tabla de Elasticidad", df_desc.to_csv(index=False).encode('utf-8'), "elasticidad.csv", "text/csv")

# ==========================================
# VISTA 3: PRICING DINÁMICO
# ==========================================
elif vista == "3. Pricing Dinámico" and df is not None:
    st.title("Pricing Dinámico y Simulación")
    
    # Filtros
    c1, c2, c3, c4 = st.columns(4)
    depto_sel = c1.selectbox("Departamento", ["Todos"] + df["dept_nm"].dropna().unique().tolist() if "dept_nm" in df.columns else ["Todos"])
    df_f = df[df["dept_nm"] == depto_sel] if depto_sel != "Todos" else df
    trim_sel = c2.selectbox("Trimestre", df_f["trimestre"].dropna().unique().tolist())
    nse_sel = c3.selectbox("NSE", ["Todos"] + df_f["categoria_est_socio"].dropna().unique().tolist())
    
    df_f = df_f[df_f["trimestre"] == trim_sel]
    if nse_sel != "Todos": df_f = df_f[df_f["categoria_est_socio"] == nse_sel]
    
    sku_sel = c4.selectbox("SKU (Unidad)", df_f["prod_nbr"].dropna().unique().tolist())
    
    st.markdown("---")
    if sku_sel:
        df_sku = df_f[df_f["prod_nbr"] == sku_sel]
        beta, _, _, _, _, _ = modelo_elasticidad(df_sku)
        el_usada = np.clip(beta, -5, 0) if pd.notna(beta) and beta < 0 else -1.0 # Fallback 
        
        # Categorización Lógica
        cat_sku = "Subir precio" if el_usada > -0.5 else ("Bajar precio" if el_usada < -1.5 else "Mantener precio")
        
        e1, e2, e3 = st.columns(3)
        esc_sel = e1.selectbox("Escenario", [e["Nombre_Escenario"] for e in ESCENARIOS])
        e2.info(f"**Categoría del SKU:** {cat_sku}")
        
        cambio = next(e["Cambio"] for e in ESCENARIOS if e["Nombre_Escenario"] == esc_sel)
        
        # Ecuación de simulación (Del Notebook Original)
        u_base = df_sku["qty"].sum()
        p_base = df_sku["precio_unitario"].mean()
        c_base = df_sku["costo_unitario"].mean()
        
        i_base = u_base * p_base
        m_base = (p_base - c_base) * u_base
        
        p_nuevo = p_base * (1 + cambio)
        u_sim = u_base * np.exp(el_usada * np.log1p(cambio))
        i_sim = p_nuevo * u_sim
        m_sim = (p_nuevo - c_base) * u_sim
        
        st.subheader("Impacto Financiero")
        k1, k2, k3 = st.columns(3)
        k1.metric("Unidades", f"{u_sim:,.0f}", f"{u_sim - u_base:,.0f} vs Base")
        k2.metric("Ingreso", f"${i_sim:,.2f}", f"${i_sim - i_base:,.2f} vs Base")
        k3.metric("Margen", f"${m_sim:,.2f}", f"${m_sim - m_base:,.2f} vs Base")
        
        # Gráficas
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(go.Figure(data=[go.Scatter(y=[i_base, i_base], name="Base"), go.Scatter(y=[i_base, i_sim], name="Simulado")]), use_container_width=True)
            st.caption("Proyección de Ventas ($).")
        with g2:
            st.plotly_chart(go.Figure(data=[go.Scatter(y=[u_base, u_base], name="Base"), go.Scatter(y=[u_base, u_sim], name="Simulado")]), use_container_width=True)
            st.caption("Proyección de Unidades.")
        with g3:
            st.plotly_chart(go.Figure(data=[go.Bar(x=["Ingreso", "Margen"], y=[i_sim, m_sim], marker_color=[COLOR_PALETTE[0], COLOR_PALETTE[3]])]), use_container_width=True)
            st.caption("Ingreso vs Margen Simulado.")

        # Conclusión
        st.success(f"💡 Para el SKU {sku_sel}, aplicar el escenario **{esc_sel}** modifica el margen en **${m_sim - m_base:,.2f}**, asumiendo una sensibilidad (elasticidad) de {el_usada:.2f}.")
        
        # Descargas
        st.markdown("---")
        experimentos = []
        mejor_esc, max_margen = None, -float('inf')
        for esc in ESCENARIOS:
            u = u_base * np.exp(el_usada * np.log1p(esc["Cambio"]))
            i = (p_base * (1 + esc["Cambio"])) * u
            m = ((p_base * (1 + esc["Cambio"])) - c_base) * u
            if m > max_margen:
                max_margen = m
                mejor_esc = esc["Nombre_Escenario"]
                
            experimentos.append({
                'SKU': sku_sel, 'dept_nm': depto_sel, 'trimestre': trim_sel, 'escenario aplicado': esc["Nombre_Escenario"],
                'unidades simuladas': u, 'ingreso simulado': i, 'margen simulado': m
            })
            
        df_exp = pd.DataFrame(experimentos)
        df_exp['mejor escenario'] = mejor_esc
        
        d1, d2 = st.columns(2)
        d1.download_button("📥 Descargar Todos los Experimentos", df_exp.to_csv(index=False).encode('utf-8'), "todos_experimentos.csv", "text/csv")
        d2.download_button("🏆 Descargar Solo el Mejor Escenario", df_exp[df_exp["escenario aplicado"] == mejor_esc].to_csv(index=False).encode('utf-8'), "mejor_escenario.csv", "text/csv")
