# Extended Drake–Information Equation UI

Interactive web app for the cosmological Extended Drake equation (v9). Run Monte Carlo sweeps over redshift z, adjust parameters like ε_waste and K_max, explore N(z) distributions, and export results. Built with Streamlit for easy data navigation in SETI research.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-orange)](https://streamlit.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://modifieddrakeequationinformation.streamlit.app)

## Purpose
Estimates N(t_0), the expected number of detectable civilizations, via cosmological integral. Key features:
- SFR fusion (MD14/lognormal + MLP uplift for mid-z peaks ~z=1.9).
- Dynamic bio/tech kernels (Deamer ramps, birth-death survival).
- Energy/detectability (fused ε_waste~0.05 from JWST nulls).
- Outputs: N_z table, log plot, CSV export.

Reproduces lit benchmarks (N_hab~10^6) with pessimistic N~10^{-13} galactic, diagnosing Fermi nulls.

## Installation
1. Clone repo: `git clone https://github.com/thomass2glitch/drake-equation-ui.git`
2. Install deps: `pip install streamlit numpy scipy pandas plotly astropy`
3. Run locally: `streamlit run app.py`

Or use the live demo above.

## Usage
- **Sliders**: Adjust z_max (0.5–3.0), ε_waste (0.01–0.5), K_max (0.8–2.0), SFR base.
- **Run Sweep**: Tap button—generates N(z) for 20 bins (dz~0.1).
- **Chat**: Type "sweep z=1.5" for dynamic runs.
- **Outputs**: Scrollable table, interactive log plot (zoom/pinch), CSV download.
- **Example**: Set ε=0.1, z_max=1.0—total N~10^{-10}, mid-z hump visible.

## Dependencies
- Python 3.9+
- streamlit==1.50.0
- numpy==1.24.0
- scipy==1.10.0
- pandas==2.0.0
- plotly==5.15.0
- astropy==6.0.0

See `requirements.txt` for pip freeze.

## Example Outputs
For defaults (z_max=2.0, ε=0.05, n=500):
- Total N: 2.15e-22 (arb units; norm ~10^{-13} galactic).
- N(z) peaks mid-z ~z=1.9 (~60% contrib).
- Plot: Log N(z) decline with hump, fade post-z=1.2.

## License
MIT—see [LICENSE](LICENSE). Free for research/education.

## Citation
[Your Name]. (2025). Extended Drake–Information Equation UI. GitHub. https://github.com/thomass2glitch/drake-equation-ui

## Contributing
Fork, PR, or open issues. Feedback on kernels/SFR fusion welcome.

## Contact
thomass2glitch@... or issues tab.
