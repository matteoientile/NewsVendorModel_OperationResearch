# numerical libraries
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, truncnorm

#visualization
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True
import seaborn as sns
import pandas as pd

####################################
######## HOW TO RUN THE CODE #######
####################################
# Simply run the entire script from top to bottom (e.g. "Run All" in your IDE).
#
# Notes:
# - All computations are vectorized via NumPy: runtime should be FAST whatever is the machine
#   even for M = 10,000 Monte Carlo scenarios.
# - 4 figures will pop up sequentially, one per analysis section (Q1 to Q4).
#   Close each figure to proceed to the next one.
# - Summary statistics are printed to the console after each figure.
####################################


####################################
######## INITIALIZATION ############
####################################

# Reproducibility
SEED = 357948
rng = default_rng(SEED)

# Initialize Economics
c = 60 #unit cost, dollars
p = 100 #unit price, dollars
r = 45 # salvage value

# Initialize distribution parameters
mu_true = 1000 # average demand
sigma_true = 150 # standard deviation


#Monte Carlo simulation parameters
n_samples = np.array([2, 5, 10, 20, 30, 50, 200, 500, 1000]) #demand data points
n_scenarios = 10_000 # number of scenarios
alpha = 0.95 #confidence level
#---

#ASSUMPTIONS
assumptions = "INITIAL ASSUMPTIONS"
print("=" * 100)
print(f"{assumptions:^100}")
print("=" * 100)
print()

print(f"{'- The product can be bought in fractions, therefore q is a positive float':<100}")
print(f"{'- Demand normally distributed':<100}")
print(f"{'- Monte Carlo replications: ' + str(n_scenarios):<100}")
print(f"{'- Unit cost (c): $' + str(c):<100}")
print(f"{'- Unit selling price (p): $' + str(p):<100}")
print(f"{'- Salvage value (r): $' + str(r):<100}")
print(f"{'- True demand mean (μ): ' + str(mu_true):<100}")
print(f"{'- True demand standard deviation (σ): ' + str(sigma_true):<100}")

print("")

####################################
############ FUNCTIONS #############
####################################

def unit_costs(c: float, p: float, r: float) -> tuple:
    """
    Compute unit costs, level of service and quantile

    Args:
        c (float): unit cost, dollars
        p (float): unit price, dollars
        r (float): salvage value

    Returns:
        c_u, c_o, los (tuple): u.c. of underage, u.c. of overage, level of service and quantile
    """
    c_u = p - c #unit cost of underage
    c_o = c - r #unit cost of overage
    los = c_u / (c_u + c_o) #level of service
    return c_u, c_o, los


def compute_qopt(mu:float, sigma: float, los: float) -> float | np.ndarray:
    """
    Compute the optimal order quantity q* = mu + sigma * z(los) to maximise Expected Profit.
    Accepts scalars (when computing exact q*) or NumPy arrays (in Monte Carlo context ) for mu and sigma 

    Args:
        mu (scalar or array): mean
        sigma (scalar or array): standard deviation
        los (float): level of service , c_u / (c_u + c_o)

    Returns:
        q_opt (float or array): optimal quantity q* or q^, depending on mu and sigma 
    """
    quantile_std = norm.ppf(los) # z(c_u / (c_u + c_o))
    q_opt = mu + sigma*quantile_std #q* or q^, i.e. the quantity that maximise the expected profit, based on the input provided
    return q_opt

def compute_ExpectedProfit(q:float, mu: float, sigma: float, c_u: float, c_o: float) -> float | np.ndarray:
    """
    Compute the true expected profit for a given quantity q.

    Accepts scalars (when computing exact E[profit] ) or NumPy arrays (in Monte Carlo context ) for q, mu and sigma

    Args:
        q (scalar or array): quantity
        mu (scalar or array): mean
        sigma (scalar or array): std
        c_u (float): u.c. of underage
        c_o (float): u.c. of overage

    Returns:
        E_profit (float): Expected Profit
    """
    inp = (q-mu)/sigma #constant val, only compute once for computational efficiency
    loss_func = norm.pdf(inp) - (inp) * (1 - norm.cdf(inp)) # loss function L

    E_profit = c_u * mu - c_o * (q-mu) - (c_u + c_o) * sigma * loss_func # expected profit
    return E_profit


def generate_scenarios(n_sample:int, n_scenarios:int, mu_true: float, sigma_true:float, rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo simulation, generating n_scenarios demand scenarios, each one having n_sample demand data points

    Args:
        n_sample (int): number of demanda data points for each scenario
        n_scenarios (int): number of scenarios
        mu_true (float): population mean
        sigma_true (float): populations std
        rng : random number generator

    Returns:
        mu_array: array of sample means, one for each scenario
        sigma_array: array of sample stds, one for each scenario
    """
    # array fulfilled with scenarios. Vectorize for computational efficiency
    scenarios_array = rng.normal(loc=mu_true, scale=sigma_true, size=(n_scenarios, n_sample))

    mu_array = np.mean(scenarios_array, axis=1) # array of sample means, one for each scenario
    sigma_array = np.std(scenarios_array, ddof=1, axis=1) #array of sample stds, one for each scenario

    return mu_array, sigma_array


def compute_EPratio(E_profit: float, ep_hats: np.ndarray) -> np.ndarray:
    """
    Compute the ratios between the Expected Profit of one scenario (i.e. considering q_hat, suboptimal) and the one computed with "q^*", the optimal quantity.

    Args:
        E_profit (float): True Expected Profit, i.e. computed with q*
        ep_hats (np.ndarray): Array of expected profits. Each entry is estimated based on q hat

    Returns:
        ep_ratios (np.ndarray): ratios E[profit(q_hat, D)] / E[profit(q*, D)]
    """
    ep_ratios = ep_hats/E_profit
    return ep_ratios


####################################
############ ANALYSIS ##############
####################################


# =======================================================================================
# Q1 - How the sample dimension affects the ratio E[profit(q^, D)]/E[profit(q*, D)]     
# =======================================================================================

# === Calculations

#Compute missing economics
c_u, c_o, los = unit_costs(c=c, p=p, r=r)

#Compute exact values
q_opt = compute_qopt(mu=mu_true, sigma=sigma_true, los=los)
ep_opt = compute_ExpectedProfit(q=q_opt, mu=mu_true, sigma=sigma_true, c_u=c_u, c_o=c_o)

print("="*100)
print(f"{'EXACT THEORETICAL VALUES (under initial fixed values)':^100}")
print("="*100)
print("")
print(f"{'- Optimal Order Quantity (q*):':<35} {q_opt:.1f} units")
print(f"{'- Maximum Expected Profit:':<35} ${ep_opt:.1f}")
print("\n")

#simulation
ratios = {}  # store n_sample -> ep_ratios array
q_suboptimals = {} #store n_sample -> q_hats
for n in n_samples:
    mu_array, sigma_array = generate_scenarios(n_sample=n, n_scenarios=n_scenarios, mu_true=mu_true, sigma_true=sigma_true, rng=rng) 

    # compute q^, E[profit(q^, D)] and the ratio E[profit(q^, D)]/E[profit(q*, D)]
    q_hats = compute_qopt(mu=mu_array, sigma=sigma_array, los=los) # shape(n_scenarios, )
    ep_hats = compute_ExpectedProfit(q=q_hats, mu=mu_true, sigma=sigma_true, c_u=c_u, c_o=c_o)
    ep_ratios = compute_EPratio(E_profit=ep_opt, ep_hats=ep_hats)

    # store results
    ratios[n] = ep_ratios
    q_suboptimals[n] = q_hats

#dfs to better handle visualizations
df_ratios=pd.DataFrame(ratios)
df_qsub = pd.DataFrame(q_suboptimals)

# === RESULTS

# PLOTS
fig, ax = plt.subplots(1,2, figsize=(15, 6))
fig.suptitle(f"Impact of Sample Size ({n_scenarios} Scenarios)", fontsize=14, y=0.97, fontweight="bold")

# boxplot 1 - q^ distribution
sns.violinplot(
    data=df_qsub,
    inner="quartile",
    ax=ax[0],
    fill=True,
    palette="Blues"
)
ax[0].axhline(y=q_opt, color='red', linestyle='--', linewidth=2, label=f'True Optimal $q^*$ ({q_opt:.1f})')

ax[0].set_title("Distribution of Estimated Order Quantity ($\hat{q}$)", fontsize=12)
ax[0].set_xlabel("Sample Size")
ax[0].set_ylabel("Order Quantity")
ax[0].legend(loc="lower right")
ax[0].grid(True, alpha=0.5)

#boxplot 2 - Ratio distribution for each Sample Size
sns.violinplot(
    data=df_ratios,
    inner="quartile",
    ax=ax[1],
    palette="Reds",
    fill=True
)
ax[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label=f'Optimality (Ratio = 1)')

ax[1].set_title(rf"Distribution of $\frac{{E[\pi(\hat{{q}}, D)]}}{{E[\pi(q^\ast, D)]}}$ Ratio", fontsize=12)
ax[1].set_xlabel("Sample Size")
ax[1].set_ylabel("Ratio")
ax[1].legend(loc="lower right")
ax[1].grid(True, alpha=0.5)


plt.tight_layout()
plt.show()

#save image in HIGH quality
#fig.savefig("q1ViolinPlots.pdf", bbox_inches="tight")

# === SUMMARY STATISTICS - Q1
print("\n" + "="*80)
print(f"{'Q1 - SUMMARY STATISTICS: Impact of Sample Size on Ratio Distribution':^80}")
print("="*80)

ci_level = alpha * 100
lower_q = (1 - alpha) / 2
upper_q = 1 - lower_q

summary_rows = [] #store results
for n in n_samples:
    rat = ratios[n]
    summary_rows.append({
        "n_sample":       n,
        "mean":           np.mean(rat),
        "std":            np.std(rat),
        "median":         np.median(rat),
        f"q{(round(lower_q*100, 1))}":  np.quantile(rat, lower_q),
        f"q{(round(upper_q*100, 1))}":  np.quantile(rat, upper_q),
        f"CI_{ci_level}%_width": np.quantile(rat, upper_q) - np.quantile(rat, lower_q),
        "pct_above_1":    100 * np.mean(rat > 1),
    })

df_summary_q1 = pd.DataFrame(summary_rows).set_index("n_sample")
print(df_summary_q1.to_string(float_format="{:.4f}".format))

# =============================================================================================
# Q2 - How the salvage price estimate error affects the ratio E[profit(q^, D)]/E[profit(q*, D)]     
# =============================================================================================
#update economics
r_true = r 
co_true = c- r_true

# === TRUNCATED NORM ASSUMPTION: r \sim TruncNorm

#truncnorm parameters
r_std = np.array([1, 2, 5, 10, 15]) #std, uncertainty on r
a_trunc = 0 #abscissa left limit, since it must be r >= 0
b_trunc = c-0.1 #abscissa right limit, since it must be r<c

#store results
store_rhats_truncnorm = {} #store the r drawned for different std
q_suboptimals_runc_truncnorm = {} #store n_sample -> q_hats
ratios_runc_truncnorm = {}  # store n_sample -> ep_ratios array

for std in r_std:
    a, b = (a_trunc - r_true) / std, (b_trunc - r_true) / std #truncnorm does not receive the plain limits as input, see docs for more info
    r_hats_tn = truncnorm.rvs(a, b, loc=r_true, scale=std, size=n_scenarios, random_state=rng) # one r for each demand scenario
    co_hats_tn = c - r_hats_tn 
    los_hats_tn = c_u/(c_u + co_hats_tn)
    q_hats_tn = compute_qopt(mu=mu_true, sigma=sigma_true, los=los_hats_tn)
    ep_hats_tn = compute_ExpectedProfit(q=q_hats_tn, mu=mu_true, sigma=sigma_true, c_u=c_u, c_o=co_true)
    ep_ratios_tn = compute_EPratio(E_profit=ep_opt, ep_hats=ep_hats_tn)

    #append results
    store_rhats_truncnorm[std] = r_hats_tn
    q_suboptimals_runc_truncnorm[std] =  q_hats_tn
    ratios_runc_truncnorm[std] = ep_ratios_tn 

#dfs to better handle visualizations
df_rhats_truncnorm = pd.DataFrame(store_rhats_truncnorm)
df_qsub_runc_truncnorm = pd.DataFrame(q_suboptimals_runc_truncnorm)
df_ratios_runc_truncnorm = pd.DataFrame(ratios_runc_truncnorm)

# === RESULTS

# PLOTS
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Impact of Salvage Value Uncertainty ({n_scenarios} Scenarios)", fontsize=14, y=0.97, fontweight="bold")

# boxplot 1 - q^ distribution
sns.violinplot(
    data=df_qsub_runc_truncnorm,
    inner="quartile",
    ax=ax[0],
    fill=True,
    palette="Blues"
)
ax[0].axhline(y=q_opt, color='red', linestyle='--', linewidth=2, label=f'True Optimal $q^*$ ({q_opt:.1f})')

ax[0].set_title("Distribution of Estimated Order Quantity ($\hat{q}$)", fontsize=12)
ax[0].set_xlabel("Salvage Value Uncertainty ($\sigma_r$)")
ax[0].set_ylabel("Order Quantity")
ax[0].legend(loc="lower right")
ax[0].grid(True, alpha=0.5)

# boxplot 2 - Ratio distribution for each Uncertainty Level
sns.violinplot(
    data=df_ratios_runc_truncnorm,
    inner="quartile",
    ax=ax[1],
    palette="Reds",
    fill=True
)
ax[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimality (Ratio = 1)')

ax[1].set_title(rf"Distribution of $\frac{{E[\pi(\hat{{q}}, D)]}}{{E[\pi(q^\ast, D)]}}$ Ratio", fontsize=12)
ax[1].set_xlabel("Salvage Value Uncertainty ($\sigma_r$)")
ax[1].set_ylabel("Ratio")
ax[1].legend(loc="lower right")
ax[1].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

#save image in HIGH quality
#fig.savefig("q2ViolinPlots.pdf", bbox_inches="tight")

# === SUMMARY STATISTICS - Q2
print("\n" + "="*80)
print(f"{'Q2 - SUMMARY STATISTICS: Impact of Salvage Value Uncertainty':^80}")
print("="*80)

summary_rows = []
for std in r_std:
    rat2 = ratios_runc_truncnorm[std]
    summary_rows.append({
        "sigma_r":        std,
        "mean":           np.mean(rat2),
        "std":            np.std(rat2),
        "median":         np.median(rat2),
        f"q{(round(lower_q*100, 1))}":  np.quantile(rat2, lower_q),
        f"q{(round(upper_q*100, 1))}":  np.quantile(rat2, upper_q),
        f"CI_{ci_level}%_width": np.quantile(rat2, upper_q) - np.quantile(rat2, lower_q),
        "pct_above_1":    100 * np.mean(rat2 > 1),
    })

df_summary_q2 = pd.DataFrame(summary_rows).set_index("sigma_r")
print(df_summary_q2.to_string(float_format="{:.4f}".format))


# =============================================================================================
# Q3 - How the ratio c_u/c_o affects the ratio E[profit(q^, D)]/E[profit(q*, D)]     
# =============================================================================================

# define the range of critical ratios (rho = c_u / c_o)
rho_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

# fix the cost and salvage value from global economics
co_fixed = c - r 

# subset of sample sizes for the heatmap
n_samples_q3 = np.array([5, 10, 30, 100, 500])

# store results
q3_ratios = [] 
q3_risk_metrics = np.zeros((len(rho_values), len(n_samples_q3))) # matrix for the heatmap

for i, rho in enumerate(rho_values):
    # update economics dynamically based on the current rho target
    cu_current = rho * co_fixed
    p_current = cu_current + c  # varying price
    los_current = cu_current / (cu_current + co_fixed)
    
    # exact optimal values for this specific margin profile
    q_opt_current = compute_qopt(mu=mu_true, sigma=sigma_true, los=los_current)
    ep_opt_current = compute_ExpectedProfit(
        q=q_opt_current, mu=mu_true, sigma=sigma_true, c_u=cu_current, c_o=co_fixed
    )
    
    for j, n in enumerate(n_samples_q3):
        # generate scenarios
        mu_array_q3, sigma_array_q3 = generate_scenarios(
            n_sample=n, n_scenarios=n_scenarios, mu_true=mu_true, sigma_true=sigma_true, rng=rng
        )
        
        # sub-optimal quantities and expected profits
        q_hats_q3 = compute_qopt(mu=mu_array_q3, sigma=sigma_array_q3, los=los_current)
        ep_hats_q3 = compute_ExpectedProfit(
            q=q_hats_q3, mu=mu_true, sigma=sigma_true, c_u=cu_current, c_o=co_fixed
        )
        ep_ratios_q3 = compute_EPratio(E_profit=ep_opt_current, ep_hats=ep_hats_q3)
        
        # extract 5th percentile for the heatmap
        q3_risk_metrics[i, j] = np.percentile(ep_ratios_q3, 5)
        
        # stre full distribution data only for n=10 for violin plot
        if n == 10:
            for ratio in ep_ratios_q3:
                q3_ratios.append({
                    'rho_label': f"{rho:.1f}",
                    'Profit Ratio': ratio
                })

# convert to pandas df for easier Seaborn plotting
df_q3_ratios = pd.DataFrame(q3_ratios)

# === RESULTS

# PLOTS
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Impact of Margin Profile ($c_u/c_o$) ({n_scenarios} Scenarios)", fontsize=14, y=0.97, fontweight="bold")

# Violin Plot showing the Tail Estimation Penalty (at n=10)
sns.violinplot(
    data=df_q3_ratios,
    x='rho_label',
    y='Profit Ratio',
    inner="quartile",
    ax=ax[0],
    palette="Reds",
    fill=True
)
ax[0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimality (Ratio = 1)')
ax[0].set_title(r"Profit Ratio Distribution ($N=10$)", fontsize=12)
ax[0].set_xlabel(r"Critical Ratio $\rho$ ($c_u/c_o$)")
ax[0].set_ylabel("Expected Profit Ratio")
ax[0].legend(loc="lower left")
ax[0].grid(True, alpha=0.5)

# Heatmap (5th Percentile)
sns.heatmap(
    q3_risk_metrics,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn", 
    xticklabels=n_samples_q3,
    yticklabels=[f"{r:.1f}" for r in rho_values],
    ax=ax[1],
    cbar_kws={'label': r'5th Percentile of $\frac{E[\pi(\hat{q})]}{E[\pi(q^\ast)]}$'}
)
ax[1].set_title("Worst 5% cases (5th Percentile of Ratio)", fontsize=12)
ax[1].set_xlabel("Sample Size ($N$)")
ax[1].set_ylabel(r"Critical Ratio $\rho$ ($c_u/c_o$)")

plt.tight_layout()
plt.show()

#save image in HIGH quality
#fig.savefig("q3ViolinHeatPlots.pdf", bbox_inches="tight")

# === SUMMARY STATISTICS - Q3 (For n=10 Violin Plot)
print("\n" + "="*80)
print(f"{'Q3 - SUMMARY STATISTICS: Impact of Margin Profile (n=10)':^80}")
print("="*80)

summary_rows_q3 = []
for rho in rho_values:
    # Filter the DataFrame for the current rho
    subset = df_q3_ratios[df_q3_ratios['rho_label'] == f"{rho:.1f}"]['Profit Ratio']
    summary_rows_q3.append({
        "rho":            rho,
        "mean":           subset.mean(),
        "std":            subset.std(),
        "median":         subset.median(),
        "q2.5":           subset.quantile(0.025),
        "q97.5":          subset.quantile(0.975),
        "CI_95.0%_width": subset.quantile(0.975) - subset.quantile(0.025),
    })

df_summary_q3 = pd.DataFrame(summary_rows_q3).set_index("rho")
print(df_summary_q3.to_string(float_format="{:.4f}".format))



# =============================================================================================
# Q4 - How the CV (sigma/mu) affects the ratio E[profit(q^, D)]/E[profit(q*, D)]     
# =============================================================================================

cv_values = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]) # range of CVs
n_samples_q4 = np.array([5, 10, 30, 100]) 

# economics kept as the initial one
c_u_q4, c_o_q4, los_q4 = unit_costs(c=c, p=p, r=r)

# store results
q4_ratios = [] # For violin plot at n=10
q4_risk_matrix = np.zeros((len(cv_values), len(n_samples_q4))) # For heatmap

for i, cv in enumerate(cv_values):
    # compute std based on CV and fixed mu
    sigma_current = cv * mu_true
    
    # optimal values
    q_opt_current = compute_qopt(mu=mu_true, sigma=sigma_current, los=los_q4)
    ep_opt_current = compute_ExpectedProfit(
        q=q_opt_current, mu=mu_true, sigma=sigma_current, c_u=c_u_q4, c_o=c_o_q4
    )
    
    for j, n in enumerate(n_samples_q4):
        # generate scenarios
        mu_array_q4, sigma_array_q4 = generate_scenarios(
            n_sample=n, n_scenarios=n_scenarios, mu_true=mu_true, sigma_true=sigma_current, rng=rng
        )
        
        # q^ and EP[profit(q^, D)]
        q_hats_q4 = compute_qopt(mu=mu_array_q4, sigma=sigma_array_q4, los=los_q4)
        ep_hats_q4 = compute_ExpectedProfit(
            q=q_hats_q4, mu=mu_true, sigma=sigma_current, c_u=c_u_q4, c_o=c_o_q4
        )
        ep_ratios_q4 = compute_EPratio(E_profit=ep_opt_current, ep_hats=ep_hats_q4)
        
        # 5th percentile for the heatmap
        q4_risk_matrix[i, j] = np.percentile(ep_ratios_q4, 5)
        
        # store data only for n=10 for violin plot
        if n == 10:
            for ratio in ep_ratios_q4:
                q4_ratios.append({
                    'CV': f"{cv:.2f}",
                    'Profit Ratio': ratio
                })

df_q4_ratios = pd.DataFrame(q4_ratios)

# === RESULTS PLOTS
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Impact of Demand Volatility ($CV = \sigma/\mu$) ({n_scenarios} Scenarios)", fontsize=14, y=0.97, fontweight="bold")

# Plot 1: violin plot (at n=10)
sns.violinplot(
    data=df_q4_ratios, x='CV', y='Profit Ratio',
    inner="quartile", ax=ax[0], palette="Reds", fill=True
)
ax[0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimality (Ratio = 1)')
ax[0].set_title(r"Profit Ratio Distribution ($N=10$)", fontsize=12)
ax[0].set_xlabel(r"Coefficient of Variation ($CV$)")
ax[0].set_ylabel("Expected Profit Ratio")
ax[0].legend(loc="lower left")
ax[0].grid(True, alpha=0.5)

# Plot 2: heatmap
sns.heatmap(
    q4_risk_matrix, annot=True, fmt=".3f", cmap="RdYlGn", 
    xticklabels=n_samples_q4, yticklabels=[f"{cv:.2f}" for cv in cv_values],
    ax=ax[1], cbar_kws={'label': r'5th Percentile of $\frac{E[\pi(\hat{q})]}{E[\pi(q^\ast)]}$'}
)
ax[1].set_title("Worst 5% cases (5th Percentile of Ratio)", fontsize=12)
ax[1].set_xlabel("Sample Size ($N$)")
ax[1].set_ylabel(r"Coefficient of Variation ($CV$)")

plt.tight_layout()
plt.show()

#save image in HIGH quality
#fig.savefig("q4ViolinHeatPlots.pdf", bbox_inches="tight")

# === SUMMARY STATISTICS - Q4
print("\n" + "="*80)
print(f"{'Q4 - SUMMARY STATISTICS: Impact of Coefficient of Variation (CV)':^80}")
print("="*80)

summary_rows_q4 = []
for cv in cv_values:
    subset = df_q4_ratios[df_q4_ratios['CV'] == f"{cv:.2f}"]['Profit Ratio']
    summary_rows_q4.append({
        "CV":             cv,
        "mean":           subset.mean(),
        "std":            subset.std(),
        "median":         subset.median(),
        "q2.5":           subset.quantile(0.025),
        "q97.5":          subset.quantile(0.975),
        "CI_95.0%_width": subset.quantile(0.975) - subset.quantile(0.025),
    })

df_summary_q4 = pd.DataFrame(summary_rows_q4).set_index("CV")
print(df_summary_q4.to_string(float_format="{:.4f}".format))