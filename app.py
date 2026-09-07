"""
QSO Absorption Line Fitter — Streamlit version.

Repo layout expected by this file:
    app.py
    requirements.txt
    fsim_HIline_all.dat
    telfer_hst.ascii
    spectra/J075547.83+220450.1.fits

The modeling functions are lifted unchanged from the Tkinter GUI; only the
interface layer is new.
"""

import ast
import io
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from linetools.analysis.voigt import voigt_from_abslines
from linetools.lists.linelist import LineList
from linetools.spectralline import AbsLine
from numpy import interp
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
SPECTRA_DIR = os.path.join(HERE, "spectra")

# Bundled example spectra: filename -> z_
EXAMPLE_SPECTRA = {
    "J075547.83+220450.1.fits": 2.3248,
}

LAM = np.arange(300, 1250, 0.01)
LAM_LL = np.arange(300, 911.76, 0.01)

st.set_page_config(page_title="Modeling Lyman Limit Systems", layout="wide")


# ----------------------------------------------------------------------
# MODELING FUNCTIONS (unchanged physics)
# ----------------------------------------------------------------------
def tau_lyman_limit(wavelength, logNHI):
    logtau = np.log10(6.3) - 18 + logNHI + 2.963 * (np.log10(wavelength) - np.log10(911.75))
    return 10.0 ** logtau


def lyman_series(HIlines, Nval, bval):
    lls_lines = []
    Nval = Nval * u.cm ** 2
    vlim = [-500.0, 500.0] * u.km / u.s
    bval = bval * u.km / u.s
    for wrest in u.Quantity(HIlines._data["wrest"]):
        aline = AbsLine(wrest, linelist=HIlines)
        aline.attrib["N"] = Nval
        aline.attrib["b"] = bval
        aline.setz(0.0)
        aline.limits.set(vlim)
        lls_lines.append(aline)
    return lls_lines


def correct_tau(tau):
    sel = ((LAM > 911.50) & (LAM < 911.66)) | ((LAM > 912.20) & (LAM < 912.40))
    interpolation = interp1d(LAM[sel], tau[sel], kind="cubic")
    fix = (LAM >= 911.66) & (LAM <= 912.20)
    tau[fix] = interpolation(LAM[fix])
    return tau


def tau_model_rest(HIlines, Nval, bval):
    tau_LL = tau_lyman_limit(LAM_LL, 17.2)
    tau_LL_rest = interp(LAM, LAM_LL, tau_LL, left=0.0, right=0.0)
    lls_lines = lyman_series(HIlines, 10 ** Nval, bval)
    tau_LS_rest = voigt_from_abslines(LAM * u.angstrom, lls_lines, ret="tau")
    return correct_tau(tau_LL_rest + tau_LS_rest)


def tau_from_column_dens(nhi):
    return (10 ** nhi) / (10 ** 17.2)


def model_continuum_spectrum(params, zqso, temp_f, temp_w):
    norm, tilt = params
    pivot_point = 950 * (1 + zqso)
    return temp_f * (temp_w / pivot_point) ** tilt * norm


def create_gaussian_lsf(fwhm, size=None):
    if size is None:
        size = int(6 * fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(-size / 2, size / 2, size)
    lsf = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return lsf / np.sum(lsf)


def convolve_spectrum_with_lsf(spec, lsf):
    return np.convolve(spec, lsf, mode="same")


def convolve_spectrum_with_lsf_var(flux, lsf_kernels, window_size_sig=9):
    convolved_spec = np.zeros_like(flux)
    window_size = window_size_sig * 17
    half_window = window_size // 2
    for i in range(len(flux)):
        start = max(0, i - half_window)
        end = min(len(flux), i + half_window + 1)
        seg = convolve_spectrum_with_lsf(flux[start:end], lsf_kernels[i])
        convolved_spec[i] = seg[half_window] if half_window < len(seg) else seg[len(seg) // 2]
    return convolved_spec


def intermediate_res(obs_wave):
    int_wave = []
    for i in range(len(obs_wave) - 1):
        int_wave.extend(np.linspace(obs_wave[i], obs_wave[i + 1], 10))
    n = len(int_wave)
    lsf_kernels = []
    for i in range(n):
        if i < 10:
            lsf_kernels.append(create_gaussian_lsf(3 * (int_wave[9] - int_wave[0])))
        elif i > n - 11:
            lsf_kernels.append(create_gaussian_lsf(3 * (int_wave[n - 1] - int_wave[n - 11])))
        else:
            lsf_kernels.append(create_gaussian_lsf(int_wave[i + 10] - int_wave[i - 10]))
    return np.array(int_wave), lsf_kernels


def absorption_model(hires_wave, t_rest, tau_abs, z_abs):
    tau_sum = np.zeros_like(hires_wave)
    for tau_i, z_i in zip(tau_abs, z_abs):
        lam_i = (1 + z_i) * LAM
        tau_sum += interp(hires_wave, lam_i, t_rest * tau_i, left=0, right=0)
    return np.exp(-tau_sum)


def parse_fit_file(content):
    data = {}
    pattern = re.compile(r"^#?\s*(\w+):\s*(.+)$", re.MULTILINE)
    for key, val in pattern.findall(content):
        val = val.strip()
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        data[key] = parsed
    for key in ["NHI", "NHI_err_low", "NHI_err_high", "z"]:
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]]
    return data


# ----------------------------------------------------------------------
# CACHED RESOURCES
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Building Lyman-series optical-depth template (one-time)...")
def load_lyman_template():
    try:
        custom_lines = Table.read(os.path.join(HERE, "fsim_HIline_all.dat"), format="ascii")
        custom_lines["wrest"] = custom_lines["wrest"] * u.AA
        custom_lines["gamma"] = custom_lines["gamma"] / u.s
        hi_lines = LineList("HI")
        hi_lines._data = custom_lines
    except Exception:
        hi_lines = LineList("HI")
    return tau_model_rest(hi_lines, 17.2, 25)


@st.cache_resource
def load_telfer():
    T02 = ascii.read(os.path.join(HERE, "telfer_hst.ascii"))
    scale = T02["flux"][T02["wrest"] == 1450.0][0]
    return np.array(T02["wrest"]), np.array(T02["flux"]) / scale, np.array(T02["err"]) / scale


@st.cache_resource(show_spinner="Loading spectrum...")
def load_spectrum(source_key, file_bytes):
    """source_key is used only as the cache key; file_bytes holds the FITS."""
    spec = Table.read(io.BytesIO(file_bytes), format="fits")
    wave = np.array(spec["wave"], dtype=float)
    flux = np.array(spec["flux"], dtype=float)
    err = np.array(spec["err"], dtype=float)
    hires_wave = np.arange(wave.min(), wave.max(), 0.01)
    int_wave, lsf_kernels = intermediate_res(wave)
    return wave, flux, err, hires_wave, int_wave, lsf_kernels


# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "absorbers" not in st.session_state:
    st.session_state.absorbers = []          # list of {"z": float, "nhi": float}
if "model" not in st.session_state:
    st.session_state.model = None
if "log" not in st.session_state:
    st.session_state.log = ["Welcome to the Lyman Limit Model Fitter."]


def log(msg):
    st.session_state.log.append(msg)


def add_absorber(z=2.0, nhi=17.2):
    st.session_state.absorbers.append({"z": float(z), "nhi": float(nhi)})


# Values loaded from a results file are applied to the sidebar widgets here,
# before those widgets are created on this run.
if "pending_params" in st.session_state:
    st.session_state["norm"], st.session_state["tilt"], st.session_state["bval"] = \
        st.session_state.pop("pending_params")


# ----------------------------------------------------------------------
# SIDEBAR — spectrum + continuum
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Spectrum")
    choice = st.selectbox("Example spectrum", list(EXAMPLE_SPECTRA.keys()))
    uploaded = st.file_uploader("...or upload your own FITS (columns: wave, flux, err)", type=["fits"])

    if uploaded is not None:
        source_key = f"upload:{uploaded.name}:{uploaded.size}"
        file_bytes = uploaded.getvalue()
        default_z = 0.0
        spec_name = uploaded.name
    else:
        source_key = f"example:{choice}"
        with open(os.path.join(SPECTRA_DIR, choice), "rb") as f:
            file_bytes = f.read()
        default_z = EXAMPLE_SPECTRA[choice]
        spec_name = choice

    zqso = st.number_input("QSO redshift", value=float(default_z), format="%.4f", step=0.001)

    st.header("Continuum")
    norm = st.number_input("Normalization", value=1.0, format="%.4f", step=0.01, key="norm")
    tilt = st.number_input("Tilt", value=0.0, format="%.4f", step=0.01, key="tilt")
    bval = st.number_input("b-parameter (km/s)", value=25.0, format="%.1f", step=1.0, key="bval")

    st.header("Plot limits")
    st.caption("Leave blank for auto.")
    c1, c2 = st.columns(2)
    x_min = c1.text_input("X min", "")
    x_max = c2.text_input("X max", "")
    y_min = c1.text_input("Y min", "")
    y_max = c2.text_input("Y max", "")

    st.header("Load fit results")
    results_file = st.file_uploader("fit_results.txt", type=["txt"], key="results")
    if results_file is not None and st.button("Apply results file"):
        try:
            data = parse_fit_file(results_file.getvalue().decode())
            missing = [k for k in ["norm", "tilt", "b", "z", "NHI"] if k not in data]
            if missing:
                log(f"Results file missing keys: {missing}")
            else:
                st.session_state.absorbers = []
                for z, n in zip(data["z"], data["NHI"]):
                    add_absorber(z, n)
                st.session_state.pending_params = (float(data["norm"]), float(data["tilt"]), float(data["b"]))
                log(f"Loaded {len(st.session_state.absorbers)} absorber(s) from {results_file.name}: "
                    f"norm={data['norm']}, tilt={data['tilt']}, b={data['b']}. Click 'Update model'.")
                st.rerun()
        except Exception as e:
            log(f"Error loading results file: {e}")


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
obs_wave, obs_flux, obs_err, hires_wave, int_wave, lsf_kernels = load_spectrum(source_key, file_bytes)
telfer_wrest, telfer_flux_rest, telfer_err_rest = load_telfer()
telfer_wave = telfer_wrest * (1 + zqso)
t_rest = load_lyman_template()

# ----------------------------------------------------------------------
# MAIN — absorbers
# ----------------------------------------------------------------------
st.title("Modeling Lyman Limit Systems")
st.caption(f"{spec_name} — z_QSO = {zqso:.4f}")

left, right = st.columns([3, 1])

with right:
    st.subheader("Absorbers")
    h1, h2, h3 = st.columns([3, 3, 1])
    h1.markdown("**Redshift**")
    h2.markdown("**log N(HI)**")
    for i, ab in enumerate(st.session_state.absorbers):
        c1, c2, c3 = st.columns([3, 3, 1])
        ab["z"] = c1.number_input("z", value=ab["z"], format="%.4f", step=0.001,
                                  key=f"z_{i}", label_visibility="collapsed")
        ab["nhi"] = c2.number_input("logN", value=ab["nhi"], format="%.3f", step=0.05,
                                    key=f"nhi_{i}", label_visibility="collapsed")
        if c3.button("✕", key=f"rm_{i}", help="Remove this absorber"):
            st.session_state.absorbers.pop(i)
            log(f"Removed absorber. {len(st.session_state.absorbers)} remaining.")
            st.rerun()

    b1, b2 = st.columns(2)
    if b1.button("Add absorber", use_container_width=True):
        add_absorber()
        log(f"Added absorber #{len(st.session_state.absorbers)}")
        st.rerun()
    if b2.button("Clear all", use_container_width=True):
        st.session_state.absorbers = []
        st.rerun()

    st.divider()
    run = st.button("Update model", type="primary", use_container_width=True)

# ----------------------------------------------------------------------
# COMPUTE MODEL
# ----------------------------------------------------------------------
if run:
    try:
        continuum = model_continuum_spectrum([norm, tilt], zqso, telfer_flux_rest, telfer_wave)
        cont_hires = interp(hires_wave, telfer_wave, continuum)
        continuum_smooth = convolve_spectrum_with_lsf(continuum, create_gaussian_lsf(28))
        continuum_spec = interp(obs_wave, telfer_wave, continuum_smooth)

        z_list = [ab["z"] for ab in st.session_state.absorbers]
        nhi_list = [ab["nhi"] for ab in st.session_state.absorbers]

        if z_list:
            tau_list = [tau_from_column_dens(n) for n in nhi_list]
            absorption = absorption_model(hires_wave, t_rest, tau_list, z_list)
            model = cont_hires * absorption
            model_int = interp(int_wave, hires_wave, model)
            model_smooth = convolve_spectrum_with_lsf_var(model_int, lsf_kernels)
            model_rebin = interp(obs_wave, int_wave, model_smooth)
        else:
            model_rebin = continuum_spec

        st.session_state.model = {
            "continuum": continuum_spec, "model": model_rebin,
            "z": z_list, "nhi": nhi_list,
            "params": {"norm": norm, "tilt": tilt, "bval": bval, "zqso": zqso},
            "spec_name": spec_name,
        }
        log(f"Model updated with {len(z_list)} absorber(s).")
    except Exception as e:
        log(f"Error updating model: {e}")

# Invalidate a cached model if the spectrum changed underneath it
m = st.session_state.model
if m is not None and m["spec_name"] != spec_name:
    m = st.session_state.model = None

# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
with left:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(obs_wave, obs_flux, "k", lw=0.8, label="Observed")
    ax.step(obs_wave, obs_err, "r", lw=0.8, alpha=0.5, label="Error")

    if m is not None:
        ax.step(obs_wave, m["continuum"], "purple", ls="--", lw=1, label="Continuum")
        ax.step(obs_wave, m["model"], "b", lw=1, label="Model")

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.legend(loc="upper right")

    def _f(s):
        try:
            return float(s)
        except ValueError:
            return None

    xlo, xhi, ylo, yhi = map(_f, (x_min, x_max, y_min, y_max))
    if xlo is not None and xhi is not None and xlo < xhi:
        ax.set_xlim(xlo, xhi)
    if ylo is not None and yhi is not None and ylo < yhi:
        ax.set_ylim(ylo, yhi)

    if m is not None:
        ymin, ymax = ax.get_ylim()
        for z, nhi in zip(m["z"], m["nhi"]):
            ll = (1 + z) * 912
            if obs_wave.min() <= ll <= obs_wave.max():
                ax.axvline(ll, color="green", ls="--", alpha=0.7)
                ax.text(ll + 10, ymax * 0.9, f"log N = {nhi:.2f}\nz = {z:.3f}",
                        rotation=90, fontsize=8, color="green", va="top")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Save model ----
    if m is not None:
        p = m["params"]
        buf = io.StringIO()
        buf.write(f"# norm: {p['norm']}\n# tilt: {p['tilt']}\n# bval: {p['bval']}\n")
        buf.write(f"# z: {m['z']}\n# NHI: {m['nhi']}\n")
        buf.write("wave\tflux\terr\tcontinuum\tmodel\n")
        for row in zip(obs_wave, obs_flux, obs_err, m["continuum"], m["model"]):
            buf.write("\t".join(f"{v:.6f}" for v in row) + "\n")
        st.download_button(
            "Download model (.txt)", buf.getvalue(),
            file_name=os.path.splitext(spec_name)[0] + "_model.txt", mime="text/plain",
        )

    with st.expander("Status log", expanded=False):
        st.text("\n".join(st.session_state.log[-30:]))
