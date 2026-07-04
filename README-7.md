# Algorithmic Options Pricing: Fourier Transform & Stochastic Simulation

An interactive quantitative finance dashboard that prices European options three different ways — closed-form Black-Scholes-Merton, the Carr-Madan Fast Fourier Transform, and Monte-Carlo simulation — and cross-checks them against each other, while also quantifying downside risk with VaR and CVaR.

---

##  Project Overview

This repository contains a full computational implementation developed as an academic project on **Algorithmic Options Pricing** at **Statmatics, IIT Kanpur**.

The goal was to build hands-on expertise in quantitative finance: deriving option prices from first principles under risk-neutral valuation, speeding up pricing with Fourier methods, and using stochastic simulation to go beyond a single price estimate into a full risk picture.

### Key Features Implemented:
* **Black-Scholes-Merton Engine:** Closed-form European option pricing under risk-neutral valuation, with the full set of Greeks — Delta, Gamma, Vega, Theta, Rho.
* **Carr-Madan FFT Pricing:** Implements the damped Fourier transform of the modified call price and maps it onto the FFT grid (with Simpson's rule weighting) to price an entire strike chain in a single `O(N log N)` pass.
* **Cross-Validation:** The FFT engine uses the GBM characteristic function, so its output is checked directly against the BSM closed-form price — the two agree to within numerical precision, confirming the damping/discretisation setup is correct.
* **Monte-Carlo Stochastic Simulation:** Simulates asset paths via the exact log-Euler scheme implied by Ito's Lemma (unbiased for any step size), giving an independent price estimate with a reported standard error.
* **Risk Analysis (VaR / CVaR):** Converts the simulated payoff distribution into a P&L distribution for a long option position and computes Value-at-Risk and Conditional VaR at a chosen confidence level.
* **Streamlit Dashboard:** Three linked tabs (BSM & Greeks, FFT Pricing, Monte-Carlo Risk) sharing the same market inputs, so all three methods are directly comparable side by side.

---

##  Mathematical Architecture

### 1. Risk-Neutral Valuation
Under the risk-neutral measure, the discounted asset price is a martingale, so a European call is priced as:

$$C(S_0,K,T) = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\max(S_T - K,\,0)]$$

Evaluating this against the lognormal density of $S_T$ (from Geometric Brownian Motion via Ito's Lemma) gives the classical closed-form BSM price.

### 2. The Damped FFT Transform
The raw call price $C_T(k)$ (as a function of log-strike $k$) is not square-integrable, so Carr & Madan (1999) price a damped call $c_T(k) \equiv e^{\alpha k}C_T(k)$ instead. Its Fourier transform is written purely in terms of the characteristic function $\phi_T(u)$:

$$\psi_T(v) = \frac{e^{-rT}\phi_T(v-(\alpha+1)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}$$

Discretising with step size $\eta$ and log-strike spacing $\lambda$, subject to $\lambda\eta = 2\pi/N$, and folding in Simpson's rule weights, gives a summation that maps exactly onto a Discrete Fourier Transform — pricing the whole strike chain in one FFT call.

### 3. Monte-Carlo via Ito's Lemma
Asset paths are simulated with the *exact* solution of the GBM SDE (no discretisation bias, even for large steps):

$$S_{t+\Delta t} = S_t \cdot \exp\!\left[(r-q-\tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right], \quad Z \sim N(0,1)$$

The discounted mean payoff across simulated paths gives the Monte-Carlo price; the simulated payoff distribution, relative to the BSM entry premium, gives the VaR / CVaR risk metrics.

---

##  Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Core Numerics:** NumPy, SciPy (`scipy.fft.fft` and `scipy.stats.norm`)
* **Data Processing:** Pandas
* **UI/Visualization:** Streamlit

To install dependencies, run:
```bash
pip install streamlit numpy pandas scipy
```

##  How to Run the Web Application

1. Clone the repository (or download `Options_Pricing_Engine.py`) to your local machine.
2. Launch the local Streamlit server:
   ```bash
   streamlit run Options_Pricing_Engine.py
   ```
3. Open your browser to the local URL (usually `http://localhost:8501`) to interact with the dashboard.

##  Using the Dashboard

The sidebar controls market inputs (spot, strike, maturity, rate, dividend yield, volatility), FFT grid settings ($N$, $\eta$, $\alpha$), and Monte-Carlo settings (number of paths, time steps, VaR/CVaR confidence). Three tabs then show:

* **Black-Scholes-Merton & Greeks** — closed-form price, Greeks table, and a price-vs-spot sensitivity curve.
* **Carr-Madan FFT Pricing** — FFT price at the chosen strike, absolute error vs BSM, and the full FFT-vs-BSM curve across the strike chain.
* **Monte-Carlo Risk Simulation** — simulated price with standard error, VaR/CVaR at the chosen confidence level, sample asset paths, the discounted payoff distribution, and a final BSM/FFT/Monte-Carlo comparison table.

##  Sample Insights & Experimentation

* Since the FFT engine uses the same GBM characteristic function as BSM, the two prices should match to within numerical precision — a useful check that the damping coefficient $\alpha$ and grid settings ($N$, $\eta$) are configured correctly.
* Increasing Monte-Carlo paths shrinks the standard error roughly as $1/\sqrt{n}$, tightening the confidence interval around the BSM/FFT benchmark.
* For a long option position, the maximum loss is capped at the premium paid, so VaR and CVaR naturally converge toward the BSM price at high confidence levels — a built-in sanity check on the risk engine.

---
