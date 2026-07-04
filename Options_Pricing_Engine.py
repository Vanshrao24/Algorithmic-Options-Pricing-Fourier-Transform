"""
=============================================================================
 ALGORITHMIC OPTIONS PRICING: FOURIER TRANSFORM & STOCHASTIC SIMULATION
=============================================================================
A quantitative finance dashboard that:
  1. Prices European options analytically under Black-Scholes-Merton (BSM)
     with full risk-neutral Greeks.
  2. Prices the SAME option using the Carr-Madan Fast Fourier Transform (FFT)
     method, validating the FFT engine against the closed-form BSM price.
  3. Runs Monte-Carlo stochastic simulations (Geometric Brownian Motion via
     Ito's Lemma discretisation) to estimate price, standard error, and
     downside risk metrics (VaR / CVaR) for an option position.

Author: Statmatics, IIT Kanpur | Algorithmic Options Pricing Project
=============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.fft import fft

st.set_page_config(layout="wide", page_title="Algorithmic Options Pricing")

# =============================================================================
# 1. BLACK-SCHOLES-MERTON: ANALYTICAL PRICE + GREEKS
# =============================================================================
def bsm_price(S0, K, r, q, T, sigma, option_type="call"):
    """Closed-form BSM price under risk-neutral valuation."""
    if T <= 0 or sigma <= 0:
        intrinsic = S0 - K if option_type == "call" else K - S0
        return max(intrinsic, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return price


def bsm_greeks(S0, K, r, q, T, sigma, option_type="call"):
    """Delta, Gamma, Vega, Theta, Rho derived from the risk-neutral density."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    gamma = (pdf_d1 * np.exp(-q * T)) / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1% vol change

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-(S0 * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S0 * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-(S0 * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S0 * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}


# =============================================================================
# 2. CARR-MADAN FFT PRICING (GBM CHARACTERISTIC FUNCTION)
# =============================================================================
def gbm_characteristic_function(u, S0, r, q, T, sigma):
    """Characteristic function of log-price sT = ln(ST) under GBM."""
    drift = np.log(S0) + (r - q - 0.5 * sigma ** 2) * T
    return np.exp(1j * u * drift - 0.5 * (sigma ** 2) * T * (u ** 2))


def carr_madan_fft(S0, r, q, T, sigma, alpha=1.5, N=4096, eta=0.25):
    """Damped Fourier-transform call pricing, mapped onto the FFT grid
    following Carr & Madan (1999). Returns a strike grid and call prices."""
    lam = (2 * np.pi) / (N * eta)
    j = np.arange(N)
    v_j = j * eta
    b = np.log(S0) - (N * lam) / 2
    k_m = b + j * lam
    strikes = np.exp(k_m)

    u = v_j - (alpha + 1) * 1j
    phi = gbm_characteristic_function(u, S0, r, q, T, sigma)
    denom = (alpha ** 2 + alpha - v_j ** 2) + 1j * (2 * alpha + 1) * v_j
    psi = (np.exp(-r * T) * phi) / denom

    weights = np.ones(N)
    weights[0] = 1 / 3
    weights[1::2] = 4 / 3
    weights[2::2] = 2 / 3
    weights[-1] = 1 / 3

    fft_input = np.exp(-1j * b * v_j) * psi * eta * weights
    fft_output = np.real(fft(fft_input))

    call_prices = (np.exp(-alpha * k_m) / np.pi) * fft_output
    return strikes, call_prices


def fft_price_at_strike(S0, K, r, q, T, sigma, alpha=1.5, N=4096, eta=0.25):
    """Interpolates the FFT strike grid to read off the price at a single K."""
    strikes, prices = carr_madan_fft(S0, r, q, T, sigma, alpha, N, eta)
    order = np.argsort(strikes)
    return np.interp(K, strikes[order], prices[order]), strikes, prices


# =============================================================================
# 3. MONTE-CARLO STOCHASTIC SIMULATION (Ito's Lemma / GBM discretisation)
# =============================================================================
def simulate_gbm_paths(S0, r, q, T, sigma, n_paths=20000, n_steps=50, seed=42):
    """Simulates asset paths under the risk-neutral GBM SDE:
        dS_t = (r - q) S_t dt + sigma S_t dW_t
    using the exact log-Euler scheme implied by Ito's Lemma:
        S_{t+dt} = S_t * exp[(r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z]
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_paths)
    paths = np.hstack([np.full((n_paths, 1), S0), paths])
    return paths


def monte_carlo_price(paths, K, r, T, option_type="call"):
    """Discounted expected payoff + standard error of the MC estimator."""
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)
    disc_payoff = np.exp(-r * T) * payoff
    price = disc_payoff.mean()
    se = disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))
    return price, se, disc_payoff


def var_cvar(pnl, confidence=0.95):
    """Historical Value-at-Risk and Conditional VaR (Expected Shortfall)
    on the simulated P&L distribution of a LONG option position."""
    losses = -pnl
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean()
    return var, cvar


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.title("Algorithmic Options Pricing: Fourier Transform & Stochastic Simulation")
st.caption("Black-Scholes-Merton \u00b7 Carr-Madan FFT \u00b7 Monte-Carlo Risk Analysis")

with st.sidebar:
    st.header("Market & Contract Inputs")
    S0 = st.number_input("Spot Price ($S_0$)", min_value=1.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price ($K$)", min_value=1.0, value=100.0, step=1.0)
    T = st.slider("Time to Maturity ($T$, years)", 0.02, 3.0, 0.5, 0.01)
    r = st.slider("Risk-Free Rate ($r$)", 0.0, 0.15, 0.05, 0.01)
    q = st.slider("Dividend Yield ($q$)", 0.0, 0.10, 0.0, 0.01)
    sigma = st.slider("Volatility ($\\sigma$)", 0.05, 0.80, 0.20, 0.01)
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)

    st.markdown("---")
    st.subheader("FFT Grid Settings")
    N_fft = st.selectbox("FFT Nodes ($N$)", [2048, 4096, 8192], index=1)
    eta = st.slider("Frequency Step ($\\eta$)", 0.05, 0.50, 0.25, 0.05)
    alpha = st.slider("Damping Coefficient ($\\alpha$)", 0.5, 3.0, 1.5, 0.1)

    st.markdown("---")
    st.subheader("Monte-Carlo Settings")
    n_paths = st.select_slider("Number of Paths", [2000, 5000, 20000, 50000], value=20000)
    n_steps = st.slider("Time Steps", 10, 250, 50, 10)
    confidence = st.slider("VaR / CVaR Confidence", 0.90, 0.99, 0.95, 0.01)

tab1, tab2, tab3 = st.tabs(
    ["Black-Scholes-Merton & Greeks", "Carr-Madan FFT Pricing", "Monte-Carlo Risk Simulation"]
)

# ---------------------------------------------------------------------------
# TAB 1: BSM ANALYTICAL PRICE + GREEKS
# ---------------------------------------------------------------------------
with tab1:
    price_bsm = bsm_price(S0, K, r, q, T, sigma, option_type)
    greeks = bsm_greeks(S0, K, r, q, T, sigma, option_type)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(f"BSM {option_type.title()} Price", f"${price_bsm:.4f}")
        st.write("**Risk-Neutral Greeks**")
        st.dataframe(pd.DataFrame(greeks, index=["Value"]).T, use_container_width=True)

    with c2:
        st.write("**Price Sensitivity to Spot Price**")
        spot_range = np.linspace(0.5 * S0, 1.5 * S0, 100)
        prices_vs_spot = [bsm_price(s, K, r, q, T, sigma, option_type) for s in spot_range]
        df_curve = pd.DataFrame({"Spot Price": spot_range, "Option Price": prices_vs_spot})
        st.line_chart(df_curve, x="Spot Price", y="Option Price")

# ---------------------------------------------------------------------------
# TAB 2: CARR-MADAN FFT PRICING (validated against BSM)
# ---------------------------------------------------------------------------
with tab2:
    fft_price, strikes, fft_prices_grid = fft_price_at_strike(
        S0, K, r, q, T, sigma, alpha=alpha, N=N_fft, eta=eta
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("BSM Analytical Price", f"${price_bsm:.4f}")
    m2.metric("Carr-Madan FFT Price", f"${fft_price:.4f}")
    m3.metric("Absolute Pricing Error", f"${abs(price_bsm - fft_price):.6f}")

    st.write("---")
    st.write("**FFT Price Curve vs BSM Closed-Form (across the strike chain)**")
    lower, upper = S0 * 0.7, S0 * 1.3
    mask = (strikes >= lower) & (strikes <= upper)
    order = np.argsort(strikes[mask])
    strikes_plot = strikes[mask][order]
    fft_plot = fft_prices_grid[mask][order]
    bsm_plot = [bsm_price(S0, k, r, q, T, sigma, option_type) for k in strikes_plot]

    df_fft = pd.DataFrame({
        "Strike": strikes_plot,
        "Carr-Madan FFT": fft_plot,
        "BSM Closed-Form": bsm_plot,
    })
    st.line_chart(df_fft, x="Strike", y=["Carr-Madan FFT", "BSM Closed-Form"])

    st.caption(
        "The whole strike chain is priced in a single O(N log N) FFT call, "
        "then read off at K by linear interpolation — versus O(N) independent "
        "closed-form evaluations for BSM. The near-zero pricing error validates "
        "the damping/discretisation setup against the analytical benchmark."
    )

# ---------------------------------------------------------------------------
# TAB 3: MONTE-CARLO STOCHASTIC SIMULATION & RISK ANALYSIS
# ---------------------------------------------------------------------------
with tab3:
    paths = simulate_gbm_paths(S0, r, q, T, sigma, n_paths=n_paths, n_steps=n_steps)
    mc_price, mc_se, disc_payoff = monte_carlo_price(paths, K, r, T, option_type)
    pnl = disc_payoff - price_bsm          # P&L of a long position vs entry premium
    var, cvar = var_cvar(pnl, confidence)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monte-Carlo Price", f"${mc_price:.4f}")
    m2.metric("Std. Error (95% CI)", f"\u00b1${1.96 * mc_se:.4f}")
    m3.metric(f"VaR ({int(confidence*100)}%)", f"${var:.4f}")
    m4.metric(f"CVaR ({int(confidence*100)}%)", f"${cvar:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Sample Simulated Asset Paths**")
        n_show = min(150, n_paths)
        time_grid = np.linspace(0, T, n_steps + 1)
        df_paths = pd.DataFrame(paths[:n_show].T, index=time_grid)
        st.line_chart(df_paths)

    with c2:
        st.write("**Discounted Payoff Distribution at Maturity**")
        hist_df = pd.DataFrame({"Discounted Payoff": disc_payoff})
        counts, bin_edges = np.histogram(disc_payoff, bins=40)
        hist_plot_df = pd.DataFrame(
            {"Payoff": bin_edges[:-1], "Frequency": counts}
        )
        st.bar_chart(hist_plot_df, x="Payoff", y="Frequency")

    st.caption(
        "Paths are simulated via the exact log-Euler scheme implied by Ito's "
        "Lemma applied to Geometric Brownian Motion. VaR/CVaR are computed on "
        "the simulated discounted P&L of a long option position relative to "
        "the BSM entry premium, at the chosen confidence level."
    )

    st.write("---")
    st.write("**Cross-Validation: BSM vs FFT vs Monte-Carlo**")
    compare_df = pd.DataFrame({
        "Method": ["Black-Scholes-Merton (closed-form)", "Carr-Madan FFT", "Monte-Carlo"],
        "Price": [price_bsm, fft_price, mc_price],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
