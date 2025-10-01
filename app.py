import streamlit as st
import numpy as np
import math
from scipy.integrate import cumtrapz, odeint
import pandas as pd
import plotly.express as px
from io import BytesIO

# App Title and Config
st.set_page_config(page_title="Extended Drake UI", layout="wide", initial_sidebar_state="expanded")
st.title("🪐 Extended Drake–Information Equation Navigator")

# Main Screen: Welcome Dashboard
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### Welcome to EDIE v9
    Estimate N(t_0), the expected number of detectable civilizations, via cosmological integral. 
    - **Key Features**: SFR fusion (MD14/lognormal + MLP for z~1.9 peak), Deamer bio ramps (mean ~0.99), birth-death survival (~0.37), fused ε_waste~0.05 from JWST nulls.
    - **Outputs**: N(z) table, log plot, CSV export.
    - **Defaults**: z_max=2.0, ε=0.05, K_max=1.1—total N~10^{-13} galactic (pessimistic realism).
    """)
with col2:
    st.markdown("""
    #### Quick Actions
    """)
    if st.button("Run Default Sweep"):
        st.session_state.run_sweep = True
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared—rerun for fresh results.")
    st.markdown("[GitHub Repo](https://github.com/thomass2glitch/drake-equation-ui) | [MIT License](LICENSE)")

# Sidebar: Controls (Always Visible)
with st.sidebar:
    st.header("Parameters")
    z_max = st.slider("z_max", 0.5, 3.0, 2.0, 0.1)
    epsilon_waste = st.slider("ε_waste (fused prior)", 0.01, 0.5, 0.05, 0.01)
    k_max = st.slider("K_max", 0.8, 2.0, 1.1, 0.1)
    sfr_base = st.selectbox("SFR Base", ["Lognormal + Rising Hybrid", "MD14 Double Power-Law"])
    n_samples = st.slider("MC Samples", 100, 1000, 500, 100)
    run_button = st.button("Run Sweep")
    chat_prompt = st.text_input("Chat Query (e.g., 'sweep z=1.5')")

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

# NumPy MLP (Scalar only)
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
    uplift
