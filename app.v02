import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="v–q (simple) • Diagrama fundamental", layout="centered")

st.title("Curva Velocidad vs Flujo (modelo simple)")

# --- Parámetros del modelo (Greenshields) ---
st.sidebar.header("Parámetros (por carril)")
vf = st.sidebar.slider("Velocidad libre Vf (km/h)", 40, 140, 100, step=5)
kj = st.sidebar.slider("Densidad de atasco kj (veh/km/carril)", 100, 220, 160, step=5)

# Densidad crítica y flujo máximo (por carril) en Greenshields
kc = kj / 2.0
vc = vf / 2.0
qmax = vf * kj / 4.0

# --- Curva v–q parametrizada por k ---
k = np.linspace(1e-6, kj, 600)              # (evita 0 para no dividir por 0)
v = vf * (1 - k / kj)                        # Greenshields: v = Vf * (1 - k/kj)
q = k * v                                    # Ecuación fundamental: q = k * v

# --- Selector de punto: "posición sobre la curva" ---
st.markdown(
    "Mueve el deslizador para **elegir un punto de la curva** (izquierda → derecha). "
    "El sistema te dirá si está **congestionado** o **no**."
)
pos = st.slider("Posición sobre la curva", 0, 100, 25)
idx = int(round(pos / 100 * (len(k) - 1)))
k_sel, v_sel, q_sel = float(k[idx]), float(v[idx]), float(q[idx])

# Clasificación: fluido vs congestionado (k visible en diagnóstico)
estado = "NO congestionado (fluido)" if k_sel <= kc else "CONGESTIONADO"

# --- Parámetros de simulación ---
st.sidebar.header("Simulación")
n_points = st.sidebar.slider("Número de puntos simulados", 20, 2000, 200, step=20)
noise_level = st.sidebar.slider("Ruido relativo en v (%)", 0, 50, 10, step=1)
regen = st.sidebar.button("Regenerar simulación")  # Forza un rerun → nueva muestra
rng = np.random.default_rng()  # Nueva semilla en cada rerun

# Generar densidades aleatorias en (0, kj]
k_sim = rng.uniform(1e-6, kj, n_points)

# Valores teóricos
v_true = vf * (1 - k_sim / kj)
q_true = k_sim * v_true

# Añadir ruido a v y recomputar q observado (consistencia física)
sigma = noise_level / 100.0
v_obs = v_true * (1 + rng.normal(0.0, sigma, n_points))
v_obs = np.clip(v_obs, 0.0, vf)           # física básica: v ≥ 0 y ≤ Vf
q_obs = k_sim * v_obs
q_obs = np.clip(q_obs, 0.0, None)         # q ≥ 0

# Máscara por régimen para colorear
mask_fluid = k_sim <= kc
mask_cong = ~mask_fluid

# --- Figura v–q ---
fig = go.Figure()

# Curva teórica
fig.add_trace(go.Scatter(
    x=q, y=v, mode="lines",
    name="Curva v–q (Greenshields)",
    line=dict(color="#1f77b4", width=3)
))

# Puntos simulados (fluido)
if np.any(mask_fluid):
    fig.add_trace(go.Scatter(
        x=q_obs[mask_fluid], y=v_obs[mask_fluid], mode="markers",
        marker=dict(size=8, color="#2ca02c", opacity=0.75),
        name="Simulado (fluido)"
    ))

# Puntos simulados (congestionado)
if np.any(mask_cong):
    fig.add_trace(go.Scatter(
        x=q_obs[mask_cong], y=v_obs[mask_cong], mode="markers",
        marker=dict(size=8, color="#ff7f0e", opacity=0.75),
        name="Simulado (congestionado)"
    ))

# Punto seleccionado por el usuario
fig.add_trace(go.Scatter(
    x=[q_sel], y=[v_sel], mode="markers+text",
    marker=dict(size=12, color="#d62728"),
    text=["Punto seleccionado"],
    textposition="top center",
    name="Punto"
))

# Pico Qmax
fig.add_vline(x=qmax, line=dict(color="gray", dash="dot"))
fig.add_annotation(x=qmax, y=vc, text="Qmax", showarrow=True, arrowhead=2, yshift=+10)

fig.update_layout(
    xaxis_title="Flujo q (veh/h/carril)",
    yaxis_title="Velocidad v (km/h)",
    height=560,
    template="plotly_white",
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig, use_container_width=True)

# --- Métricas y diagnóstico (modelo) ---
col1, col2, col3 = st.columns(3)
col1.metric("Qmax (por carril)", f"{qmax:,.0f} veh/h")
col2.metric("kc (densidad crítica)", f"{kc:,.1f} veh/km/carril")
col3.metric("v@Qmax", f"{vc:,.1f} km/h")

st.subheader("Diagnóstico del punto seleccionado")
st.markdown(f"""
- **Resultado:** **{estado}**
- **Valores del punto:**  
  • Flujo **q** = **{q_sel:,.0f}** veh/h/carril  
  • Velocidad **v** = **{v_sel:,.1f}** km/h  
  • (Interno) Densidad inferida **k = q/v** = **{k_sel:,.1f}** veh/km/carril  
- **Criterio**: si \\(k \\le k_c = k_j/2\\) ⇒ **fluido**; si \\(k > k_c\\) ⇒ **congestionado**.
""")

# --- Calidad del ajuste de la simulación respecto a la curva ---
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

rmse_v = float(np.sqrt(np.mean((v_obs - v_true) ** 2)))
rmse_q = float(np.sqrt(np.mean((q_obs - q_true) ** 2)))
r2_v = float(r2_score(v_true, v_obs))
r2_q = float(r2_score(q_true, q_obs))

st.subheader("Calidad de la simulación vs. curva teórica")
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE(v)", f"{rmse_v:,.2f} km/h")
c2.metric("RMSE(q)", f"{rmse_q:,.0f} veh/h")
c3.metric("R²(v)", f"{r2_v:,.3f}")
c4.metric("R²(q)", f"{r2_q:,.3f}")

# --- Descargar datos simulados ---
sim_df = pd.DataFrame({
    "k (veh/km/carril)": k_sim,
    "v_true (km/h)": v_true,
    "q_true (veh/h/carril)": q_true,
    "v_obs (km/h)": v_obs,
    "q_obs (veh/h/carril)": q_obs,
    "regimen": np.where(mask_fluid, "fluido", "congestionado")
})
st.download_button(
    "Descargar datos simulados (CSV)",
    sim_df.to_csv(index=False).encode("utf-8"),
    file_name="simulacion_vq.csv",
    mime="text/csv"
)

st.caption(
    "Notas: La simulación muestrea densidades k en [0, kj] y aplica ruido relativo a la velocidad; "
    "el flujo observado se recalcula como q = k·v_obs para mantener consistencia física. "
    "Si los datos provienen de un tramo que obedece Greenshields bajo estos parámetros Vf y kj, "
    "los puntos se distribuirán alrededor de la curva. Ajusta el ruido para explorar distintas dispersiones."
)
