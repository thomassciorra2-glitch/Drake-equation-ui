import streamlit as st
import numpy as np
import math  # ← FIX 1: Added missing import
from scipy.integrate import cumtrapz, odeint
import pandas as pd
import plotly.express as px
from io import BytesIO

# App Title
st.set_page_config(page_title="Extended Drake UI", layout="wide", initial_sidebar_state="expanded")
st.title("🪐 Extended Drake–Information Equation Navigator")
st.markdown("Interactive MC sweeps for N(t_0). Adjust params, run, explore N(z).")

# Sidebar: Params & Controls
st.sidebar.header("Parameters")
z_max = st.sidebar.slider("z_max", 0.5, 3.0, 2.0, 0.1)
epsilon_waste = st.sidebar.slider("ε_waste (fused prior)", 0.01, 0.5, 0.05, 0.01)
k_max = st.sidebar.slider("K_max", 0.8, 2.0, 1.1, 0.1)
sfr_base = st.sidebar.selectbox("SFR Base", ["Lognormal + Rising Hybrid", "MD14 Double Power-Law"])
n_samples = st.sidebar.slider("MC Samples", 100, 1000, 500, 100)
run_button = st.sidebar.button("Run Sweep")

# Chat Input
chat_prompt = st.sidebar.text_input("Chat Query (e.g., 'sweep z=1.5')")

# Cosmology Fallback
H0 = 70.0 * 1000.0 / 3.085677581e22
Om0, Ol0 = 0.3, 0.7
c = 299792458.0

def Ez(z): return math.sqrt(Om0*(1+z)**3 + Ol0)
def dc_comoving(z, steps=1024):
    zgrid = np.linspace(0.0, z, steps)
    integrand = 1.0 / np.array([Ez(zz) for zz in zgrid])
    return (c / H0) * np.trapz(integrand, zgrid)
def dL(z): return (1+z) * dc_comoving(z)

# Hard-Steps Bio
def B_hardsteps(t_star, r0=0.1, tau=0.05, alpha=2):
    if np.isscalar(t_star):
        t_star = np.array([t_star])
    t_grid = np.linspace(0, t_star.max(), 100)
    lambda_chem = r0 * t_grid / (1 + (t_grid / tau)**alpha)
    integral = cumtrapz(lambda_chem, t_grid, initial=0)
    B_interp = np.interp(t_star, t_grid, integral)
    B = 1 - np.exp(-B_interp)
    return B.mean() if len(B) > 1 else B[0]

# Birth-Death Cultural
def C_birthdeath(t_star, b=0.05, d=0.03, K_cap=1e16):
    if np.isscalar(t_star):
        t_star = np.array([t_star])
    def ode(y, t):
        return b * y - d * y**2 / K_cap
    y0 = 1.0
    sol = odeint(ode, y0, t_star)
    return sol[:, 0].mean() / K_cap

# NumPy MLP (FIX 2: Simple scalar, no array waste)
def numpy_mlp(z):
    x = np.array([z])
    h = 1 / (1 + np.exp(- (x - 0.5) * 10))
    out = 1 / (1 + np.exp(-h * 10))
    return out[0]

# SFR Bases
def md14_sfr(z, phi0=0.01, alpha_low=-0.3, alpha_high=-3.5, beta=1.5):
    return phi0 * (1 + z)**alpha_low / ((1 + z)**alpha_low + (1 + z)**alpha_high)**beta

def lognormal_rising_sfr(z, phi0=0.01, mu=0.64, sigma=0.5, gamma=0.5, z0=1.5):
    if z == 0:
        z = 1e-6
    lognorm = phi0 / (z * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(z) - mu)**2 / (2 * sigma**2) )
    rising = (1 + z)**gamma * np.exp(-z / z0)
    return lognorm * rising

def psi_z(z, sfr_type='Lognormal + Rising Hybrid'):
    uplift = numpy_mlp(z)
    uplift_factor = 1.0 + 0.5 * uplift  # [1.0, 1.5]
    if sfr_type == 'MD14 Double Power-Law':
        base = md14_sfr(z)
    else:
        base = lognormal_rising_sfr(z)
    return base * uplift_factor

# Full MC Function (FIX 3: @st.cache_resource for MLP, scalar random)
@st.cache_data
def drake_mc(z_max, n_samples, epsilon_waste, k_max, sfr_base):
    z_bins = np.linspace(0.05, z_max, 20)
    n_z = np.zeros(len(z_bins))
    K_cap = 10**(10 * k_max + 6)
    for i, z in enumerate(z_bins):
        t_star = 10.0 - z * 8.0
        psi = psi_z(z, sfr_base) * n_samples
        B = B_hardsteps(t_star)
        C = C_birthdeath(t_star, K_cap=K_cap)
        p_det = epsilon_waste / (1 + z)**4 * np.random.uniform(0.01, 0.1, size=1).mean()  # FIX: Scalar, no array
        n_z[i] = psi * B * C * p_det
    total_n = np.sum(n_z)
    df = pd.DataFrame({'z': z_bins, 'N_z': n_z, 'Total N': total_n})
    return df, total_n

# Main App Logic
if run_button or chat_prompt:
    if chat_prompt:
        try:
            z_max = float(chat_prompt.split('z=')[1].split()[0]) if 'z=' in chat_prompt else z_max
        except:
            pass
    df, total_n = drake_mc(z_max, n_samples, epsilon_waste, k_max, sfr_base)
    
    if total_n > 0:
        st.subheader(f"Results: N(z) for z_max={z_max}")
        st.dataframe(df, use_container_width=True)
        
        fig = px.line(df, x='z', y='N_z', log_y=True, title=f'log N(z) - Total N = {total_n:.2e}')
        st.plotly_chart(fig, use_container_width=True)
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("Download CSV", csv_buffer.getvalue(), "n_z_results.csv", "text/csv")
    else:
        st.error("Run failed—check params.")

# Footer
st.markdown("---")
st.caption("Prototype for EDIE v9. Open-source under MIT. Run sweeps, explore data.")
