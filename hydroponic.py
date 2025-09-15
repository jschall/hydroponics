import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse

# Constants
RESERVOIR_VOLUME = 4.0  # liters
AIR_TEMPERATURE = 85.0  # Fahrenheit
HUMIDITY = 0.7  # 70% relative humidity
HUMIDITY_SCALE = 0.8  # 20% reduction for low VPD (85F, 70% RH)
HUMIDITY_SCALE_STD = 0.1  # Variance for humidity scale (absolute)
HUMIDITY_STD = 0.2  # Absolute RH fraction std (e.g., 0.05 = 5% RH)
LIGHT_DLI_BASE = 25.0  # mol/m^2/day reference DLI
LIGHT_DLI = 25.0  # default DLI
LIGHT_DLI_STD = 15.0  # std for DLI
AIR_TEMPERATURE_STD = 5.0  # Fahrenheit std
TEMP_Q10 = 2.0  # Q10 temperature factor for uptake scaling
RISK_LAMBDA = 1.0  # weight on variance in expected loss
N_A = 60.0  # ppm N per ml/L, Part A (6% N)
N_B = 10.0  # ppm N per ml/L, Part B (1% N)
P_A = 0.0  # ppm P per ml/L, Part A
P_B = 21.8  # ppm P per ml/L, Part B (5% P2O5 * 0.436)
K_A = 41.5  # ppm K per ml/L, Part A (5% K2O * 0.83)
K_B = 49.8  # ppm K per ml/L, Part B (6% K2O * 0.83)
CA_A = 50.0  # ppm Ca per ml/L, Part A (5% Ca)
CA_B = 0.0   # ppm Ca per ml/L, Part B
MG_A = 0.0   # ppm Mg per ml/L, Part A
MG_B = 12.0  # ppm Mg per ml/L, Part B (1.2% Mg)
FE_A = 1.2   # ppm Fe per ml/L, Part A (0.12% Fe chelated)
FE_B = 0.0   # ppm Fe per ml/L, Part B
SIM_DAYS = 120  # simulation duration
N_TO_BIOMASS = 0.02  # g biomass per mg N uptake (approx., peppers ~2-3% N by dry weight)
N_TO_BIOMASS_STD = 0.004  # Variance for biomass conversion
OPT_WEIGHTS = {'N': 3.0, 'P': 2.0, 'K': 2.0, 'Ca': 1.5, 'Mg': 1.0, 'Fe': 0.5}  # Weights by nutrient importance (macros higher)

# Michaelis-Menten parameters (V_max in mg/day/2 plants, K_m in ppm) with std devs for variance
N_V_MAX = 74.0  # max N uptake at peak (~37 mg/plant/day)
N_V_MAX_STD = 14.8  # 20% std
P_V_MAX = 34.0  # max P uptake at peak (~17 mg/plant/day)
P_V_MAX_STD = 6.8  # 20% std
K_V_MAX = 116.0  # max K uptake at peak (~58 mg/plant/day)
K_V_MAX_STD = 23.2  # 20% std
CA_V_MAX = 24.4  # max Ca uptake at peak (~12.2 mg/plant/day)
CA_V_MAX_STD = 4.88  # 20% std
MG_V_MAX = 11.6  # max Mg uptake at peak (~5.8 mg/plant/day)
MG_V_MAX_STD = 2.32  # 20% std
FE_V_MAX = 0.62  # max Fe uptake at peak (~0.31 mg/plant/day)
FE_V_MAX_STD = 0.124  # 20% std
N_K_M = 100.0    # half-saturation for N (~50% target 200 ppm)
N_K_M_STD = 20.0  # 20% std
P_K_M = 30.0     # half-saturation for P (~50% target 60 ppm)
P_K_M_STD = 6.0  # 20% std
K_K_M = 135.0    # half-saturation for K (~50% target 270 ppm)
K_K_M_STD = 27.0  # 20% std
CA_K_M = 85.0    # half-saturation for Ca (~50% target 170 ppm)
CA_K_M_STD = 17.0  # 20% std
MG_K_M = 28.0    # half-saturation for Mg (~50% target 56 ppm)
MG_K_M_STD = 5.6  # 20% std
FE_K_M = 1.5     # half-saturation for Fe (~50% target 3 ppm)
FE_K_M_STD = 0.3  # 20% std

# Note: Model includes concentration-dependent uptake with variances on uncertain params (V_max, K_m, humidity scale, biomass conv.).
# Optimization weighted by nutrient importance. Plots show mean +/- 1 std shading from Monte Carlo sims.

# Nutrient parameter table for DRY uptake functions
NUTRIENT_PARAMS = {
    'N': {'v_max': N_V_MAX, 'k_m': N_K_M},
    'P': {'v_max': P_V_MAX, 'k_m': P_K_M},
    'K': {'v_max': K_V_MAX, 'k_m': K_K_M},
    'Ca': {'v_max': CA_V_MAX, 'k_m': CA_K_M},
    'Mg': {'v_max': MG_V_MAX, 'k_m': MG_K_M},
    'Fe': {'v_max': FE_V_MAX, 'k_m': FE_K_M},
}

# Define daily water uptake for 2 Thai pepper plants (L/day total)
def daily_water(d, scale=HUMIDITY_SCALE, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    # Temperature scaling via Q10 rule (reference 77F)
    q10_factor = TEMP_Q10 ** ((temp_f - 77.0) / 18.0)
    if d < 30:
        w_per_plant = 0.005 * d  # peaks at 0.15 L/plant/day
    elif d < 60:
        w_per_plant = 0.15 + 0.008 * (d - 30)  # peaks at ~0.39 L/plant/day
    else:
        w_per_plant = 0.39 + 0.001 * (d - 60)  # to ~0.5 L/plant/day
    return q10_factor * scale * light_factor * 2 * w_per_plant  # max ~0.8 L/day total

# Concentration-dependent uptake function (mg/day/2 plants)
def uptake(c, d, v_max, k_m, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    q10_factor = TEMP_Q10 ** ((temp_f - 77.0) / 18.0)
    v_max_light = v_max * light_factor * q10_factor
    v_max_adj = v_max_light * (0.5 + 0.5 * d / total_days)  # Gradual increase to V_MAX
    return v_max_adj * c / (k_m + c)

# Specific uptake functions for each nutrient
def uptake_n(c, d, v_max=N_V_MAX, k_m=N_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

def uptake_p(c, d, v_max=P_V_MAX, k_m=P_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

def uptake_k(c, d, v_max=K_V_MAX, k_m=K_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

def uptake_ca(c, d, v_max=CA_V_MAX, k_m=CA_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

def uptake_mg(c, d, v_max=MG_V_MAX, k_m=MG_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

def uptake_fe(c, d, v_max=FE_V_MAX, k_m=FE_K_M, total_days=SIM_DAYS, light_factor=1.0, temp_f=AIR_TEMPERATURE):
    return uptake(c, d, v_max, k_m, total_days, light_factor, temp_f)

# Target concentrations (ppm elemental)
def target_n_ppm(d):
    if d < 30:
        return 50 + 2 * d  # 50 to 110 ppm N
    elif d < 60:
        return 110 + 3 * (d - 30)  # 110 to 200 ppm N
    else:
        return 200 + 0.5 * (d - 60)  # 200 to 260 ppm N

def target_p_ppm(d):
    if d < 30:
        return 20 + 0.67 * d  # 20 to 40 ppm P
    elif d < 60:
        return 40 + 0.67 * (d - 30)  # 40 to 60 ppm P
    else:
        return 60 + 0.17 * (d - 60)  # 60 to 70 ppm P

def target_k_ppm(d):
    if d < 30:
        return 50 + 1.5 * d  # 50 to 95 ppm K
    elif d < 60:
        return 95 + 2 * (d - 30)  # 95 to 175 ppm K (continuous)
    else:
        return 175 + 1 * (d - 60)  # 175 to 235 ppm K

def target_ca_ppm(d):
    if d < 30:
        return 50 + 1 * d  # 50 to 80 ppm Ca
    elif d < 60:
        return 80 + 2 * (d - 30)  # 80 to 140 ppm Ca
    else:
        return 140 + 0.5 * (d - 60)  # 140 to 170 ppm Ca

def target_mg_ppm(d):
    if d < 30:
        return 20 + 0.5 * d  # 20 to 35 ppm Mg
    elif d < 60:
        return 35 + 0.5 * (d - 30)  # 35 to 50 ppm Mg
    else:
        return 50 + 0.1 * (d - 60)  # 50 to 56 ppm Mg

def target_fe_ppm(d):
    if d < 30:
        return 1 + 0.033 * d  # 1 to 2 ppm Fe
    elif d < 60:
        return 2 + 0.033 * (d - 30)  # 2 to 3 ppm Fe
    else:
        return 3  # steady at 3 ppm Fe

# Simulation function: evolve concentrations with concentration-dependent uptake and param variance
def simulate(ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days=SIM_DAYS, sample_params=None, light_dli=LIGHT_DLI, air_temp_f=AIR_TEMPERATURE, humidity=HUMIDITY):
    if sample_params is None:
        sample_params = {}

    # Deterministic mean parameter values with optional overrides
    humidity_scale = sample_params.get('humidity_scale', HUMIDITY_SCALE)
    n_v_max = sample_params.get('n_v_max', N_V_MAX)
    p_v_max = sample_params.get('p_v_max', P_V_MAX)
    k_v_max = sample_params.get('k_v_max', K_V_MAX)
    ca_v_max = sample_params.get('ca_v_max', CA_V_MAX)
    mg_v_max = sample_params.get('mg_v_max', MG_V_MAX)
    fe_v_max = sample_params.get('fe_v_max', FE_V_MAX)
    n_k_m = sample_params.get('n_k_m', N_K_M)
    p_k_m = sample_params.get('p_k_m', P_K_M)
    k_k_m = sample_params.get('k_k_m', K_K_M)
    ca_k_m = sample_params.get('ca_k_m', CA_K_M)
    mg_k_m = sample_params.get('mg_k_m', MG_K_M)
    fe_k_m = sample_params.get('fe_k_m', FE_K_M)
    n_to_biomass = sample_params.get('n_to_biomass', N_TO_BIOMASS)

    light_factor = max(0.2, light_dli / LIGHT_DLI_BASE)

    n_ppm = np.zeros(days + 1)
    p_ppm = np.zeros(days + 1)
    k_ppm = np.zeros(days + 1)
    ca_ppm = np.zeros(days + 1)
    mg_ppm = np.zeros(days + 1)
    fe_ppm = np.zeros(days + 1)
    n_uptake_total = np.zeros(days + 1)

    # Initial concentrations (ppm)
    n_ppm[0] = N_A * ml_a_init + N_B * ml_b_init
    p_ppm[0] = P_A * ml_a_init + P_B * ml_b_init
    k_ppm[0] = K_A * ml_a_init + K_B * ml_b_init
    ca_ppm[0] = CA_A * ml_a_init + CA_B * ml_b_init
    mg_ppm[0] = MG_A * ml_a_init + MG_B * ml_b_init
    fe_ppm[0] = FE_A * ml_a_init + FE_B * ml_b_init

    for d in range(days):
        W = daily_water(d, humidity_scale, light_factor, air_temp_f)

        # Concentration-dependent uptake with sampled params
        U_n = uptake_n(n_ppm[d], d, n_v_max, n_k_m, days, light_factor, air_temp_f)
        U_p = uptake_p(p_ppm[d], d, p_v_max, p_k_m, days, light_factor, air_temp_f)
        U_k = uptake_k(k_ppm[d], d, k_v_max, k_k_m, days, light_factor, air_temp_f)
        U_ca = uptake_ca(ca_ppm[d], d, ca_v_max, ca_k_m, days, light_factor, air_temp_f)
        U_mg = uptake_mg(mg_ppm[d], d, mg_v_max, mg_k_m, days, light_factor, air_temp_f)
        U_fe = uptake_fe(fe_ppm[d], d, fe_v_max, fe_k_m, days, light_factor, air_temp_f)

        # Cap by available mass
        U_n_actual = min(U_n, n_ppm[d] * RESERVOIR_VOLUME)
        n_uptake_total[d + 1] = n_uptake_total[d] + U_n_actual
        mass_n_after = n_ppm[d] * RESERVOIR_VOLUME - U_n_actual
        n_mass_new = mass_n_after + (N_A * ml_a_refill + N_B * ml_b_refill) * W
        n_ppm[d + 1] = max(0, n_mass_new / RESERVOIR_VOLUME)

        U_p_actual = min(U_p, p_ppm[d] * RESERVOIR_VOLUME)
        mass_p_after = p_ppm[d] * RESERVOIR_VOLUME - U_p_actual
        p_mass_new = mass_p_after + (P_A * ml_a_refill + P_B * ml_b_refill) * W
        p_ppm[d + 1] = max(0, p_mass_new / RESERVOIR_VOLUME)

        U_k_actual = min(U_k, k_ppm[d] * RESERVOIR_VOLUME)
        mass_k_after = k_ppm[d] * RESERVOIR_VOLUME - U_k_actual
        k_mass_new = mass_k_after + (K_A * ml_a_refill + K_B * ml_b_refill) * W
        k_ppm[d + 1] = max(0, k_mass_new / RESERVOIR_VOLUME)

        U_ca_actual = min(U_ca, ca_ppm[d] * RESERVOIR_VOLUME)
        mass_ca_after = ca_ppm[d] * RESERVOIR_VOLUME - U_ca_actual
        ca_mass_new = mass_ca_after + (CA_A * ml_a_refill + CA_B * ml_b_refill) * W
        ca_ppm[d + 1] = max(0, ca_mass_new / RESERVOIR_VOLUME)

        U_mg_actual = min(U_mg, mg_ppm[d] * RESERVOIR_VOLUME)
        mass_mg_after = mg_ppm[d] * RESERVOIR_VOLUME - U_mg_actual
        mg_mass_new = mass_mg_after + (MG_A * ml_a_refill + MG_B * ml_b_refill) * W
        mg_ppm[d + 1] = max(0, mg_mass_new / RESERVOIR_VOLUME)

        U_fe_actual = min(U_fe, fe_ppm[d] * RESERVOIR_VOLUME)
        mass_fe_after = fe_ppm[d] * RESERVOIR_VOLUME - U_fe_actual
        fe_mass_new = mass_fe_after + (FE_A * ml_a_refill + FE_B * ml_b_refill) * W
        fe_ppm[d + 1] = max(0, fe_mass_new / RESERVOIR_VOLUME)

    cumulative_mass = n_uptake_total * n_to_biomass / 2  # g dry weight per plant (2 plants)

    return n_ppm, p_ppm, k_ppm, ca_ppm, mg_ppm, fe_ppm, n_uptake_total, cumulative_mass

# For initial reservoir: optimize for average over first 30 days (weighted MSE)
def objective_init(x):
    ml_a, ml_b = x
    target_n_avg = np.mean([target_n_ppm(d) for d in range(30)])
    target_p_avg = np.mean([target_p_ppm(d) for d in range(30)])
    target_k_avg = np.mean([target_k_ppm(d) for d in range(30)])
    target_ca_avg = np.mean([target_ca_ppm(d) for d in range(30)])
    target_mg_avg = np.mean([target_mg_ppm(d) for d in range(30)])
    target_fe_avg = np.mean([target_fe_ppm(d) for d in range(30)])
    n = N_A * ml_a + N_B * ml_b
    p = P_A * ml_a + P_B * ml_b
    k = K_A * ml_a + K_B * ml_b
    ca = CA_A * ml_a + CA_B * ml_b
    mg = MG_A * ml_a + MG_B * ml_b
    fe = FE_A * ml_a + FE_B * ml_b
    mse_n = OPT_WEIGHTS['N'] * (n - target_n_avg)**2
    mse_p = OPT_WEIGHTS['P'] * (p - target_p_avg)**2
    mse_k = OPT_WEIGHTS['K'] * (k - target_k_avg)**2
    mse_ca = OPT_WEIGHTS['Ca'] * (ca - target_ca_avg)**2
    mse_mg = OPT_WEIGHTS['Mg'] * (mg - target_mg_avg)**2
    mse_fe = OPT_WEIGHTS['Fe'] * (fe - target_fe_avg)**2
    return mse_n + mse_p + mse_k + mse_ca + mse_mg + mse_fe

bounds = [(0, 10), (0, 10)]
result_init = minimize(objective_init, [1.0, 1.0], bounds=bounds, method='SLSQP')
ml_a_init, ml_b_init = result_init.x

# Objective for refill: probabilistic health with analytical uncertainty
def objective_refill(x, sim_days=SIM_DAYS, light_dli=LIGHT_DLI, air_temp_f=AIR_TEMPERATURE, humidity=HUMIDITY):
    ml_a_refill, ml_b_refill = x
    target_n = np.array([target_n_ppm(d) for d in range(sim_days + 1)])
    target_p = np.array([target_p_ppm(d) for d in range(sim_days + 1)])
    target_k = np.array([target_k_ppm(d) for d in range(sim_days + 1)])
    target_ca = np.array([target_ca_ppm(d) for d in range(sim_days + 1)])
    target_mg = np.array([target_mg_ppm(d) for d in range(sim_days + 1)])
    target_fe = np.array([target_fe_ppm(d) for d in range(sim_days + 1)])
    # Deterministic mean trajectory
    sample_params = {
        'humidity_scale': HUMIDITY_SCALE,
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    n_sim, p_sim, k_sim, ca_sim, mg_sim, fe_sim, _, _ = simulate(
        ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days=sim_days, sample_params=sample_params, light_dli=light_dli, air_temp_f=air_temp_f, humidity=humidity
    )

    # Sensitivity-weighted expected squared error + variance term (delta method)
    # Approximate variance from parameter stds via partial contributions (independent assumption)
    # For each nutrient, variance ~ (dU/dvmax * sigma_vmax)^2 at mean, aggregated over days into concentration
    # We approximate with proportionality to squared light-adjusted uptake fraction
    def approx_var(conc_series, v_max_std):
        # Scale with concentration and growth stage; normalized heuristic
        stage = np.linspace(0.5, 1.0, len(conc_series))
        scale = (conc_series / (conc_series + 1.0))**2 * stage**2
        return (v_max_std**2) * scale

    # Add climate-driven variance approximations
    light_var = (LIGHT_DLI_STD / max(1e-6, LIGHT_DLI_BASE))**2
    temp_var = (AIR_TEMPERATURE_STD / 18.0 * np.log(TEMP_Q10))**2
    humidity_var = HUMIDITY_SCALE_STD**2

    # Sensitivity multipliers by nutrient to reflect plant health sensitivity
    SENS = OPT_WEIGHTS  # reuse weights as sensitivities by default

    n_var = approx_var(n_sim, N_V_MAX_STD)
    p_var = approx_var(p_sim, P_V_MAX_STD)
    k_var = approx_var(k_sim, K_V_MAX_STD)
    ca_var = approx_var(ca_sim, CA_V_MAX_STD)
    mg_var = approx_var(mg_sim, MG_V_MAX_STD)
    fe_var = approx_var(fe_sim, FE_V_MAX_STD)

    climate_var_scale = light_var + temp_var + humidity_var
    loss_n = SENS['N'] * np.mean((n_sim - target_n)**2 + RISK_LAMBDA * (n_var + climate_var_scale * (target_n**0)))
    loss_p = SENS['P'] * np.mean((p_sim - target_p)**2 + RISK_LAMBDA * (p_var + climate_var_scale * (target_p**0)))
    loss_k = SENS['K'] * np.mean((k_sim - target_k)**2 + RISK_LAMBDA * (k_var + climate_var_scale * (target_k**0)))
    loss_ca = SENS['Ca'] * np.mean((ca_sim - target_ca)**2 + RISK_LAMBDA * (ca_var + climate_var_scale * (target_ca**0)))
    loss_mg = SENS['Mg'] * np.mean((mg_sim - target_mg)**2 + RISK_LAMBDA * (mg_var + climate_var_scale * (target_mg**0)))
    loss_fe = SENS['Fe'] * np.mean((fe_sim - target_fe)**2 + RISK_LAMBDA * (fe_var + climate_var_scale * (target_fe**0)))

    return loss_n + loss_p + loss_k + loss_ca + loss_mg + loss_fe

def main():
    args = parse_args()

    # Override global constants with CLI args
    global SIM_DAYS
    SIM_DAYS = args.days
    light_dli = args.light_dli
    air_temp = args.air_temp_f
    rh = args.humidity

    # Optimize refill concentrations
    print("Optimizing refill concentrations for Parts A and B...")
    initial_guess = [1.0, 1.0]
    # Update global risk lambda from CLI
    global RISK_LAMBDA
    RISK_LAMBDA = args.risk_lambda

    result = minimize(lambda x: objective_refill(x, SIM_DAYS, light_dli, air_temp, rh), initial_guess, bounds=bounds, method='SLSQP')
    optimal_ml_a_refill, optimal_ml_b_refill = result.x
    optimal_n_refill = N_A * optimal_ml_a_refill + N_B * optimal_ml_b_refill
    optimal_p_refill = P_B * optimal_ml_b_refill
    optimal_k_refill = K_A * optimal_ml_a_refill + K_B * optimal_ml_b_refill
    optimal_ca_refill = CA_A * optimal_ml_a_refill + CA_B * optimal_ml_b_refill
    optimal_mg_refill = MG_A * optimal_ml_a_refill + MG_B * optimal_ml_b_refill
    optimal_fe_refill = FE_A * optimal_ml_a_refill + FE_B * optimal_ml_b_refill

    # Scale to 4L for initial
    ml_a_init_4l = ml_a_init * RESERVOIR_VOLUME
    ml_b_init_4l = ml_b_init * RESERVOIR_VOLUME

    print("\nOptimization Results (for ENVY Parts A (6-0-5) and B (1-5-6)):")
    print(f"Initial in 4L reservoir (higher A:B for seedlings): {ml_a_init_4l:.2f} ml Part A, {ml_b_init_4l:.2f} ml Part B")
    print(f"  Ratio A:B = {ml_a_init:.2f}:{ml_b_init:.2f}")
    print(f"  Achieves approx. N {N_A * ml_a_init + N_B * ml_b_init:.1f} ppm, P {P_B * ml_b_init:.1f} ppm, K {K_A * ml_a_init + K_B * ml_b_init:.1f} ppm")
    print(f"  Ca {CA_A * ml_a_init + CA_B * ml_b_init:.1f} ppm, Mg {MG_A * ml_a_init + MG_B * ml_b_init:.1f} ppm, Fe {FE_A * ml_a_init + FE_B * ml_b_init:.1f} ppm")
    print(f"Optimal for refill reservoir (balanced for full cycle): {optimal_ml_a_refill:.2f} ml Part A and {optimal_ml_b_refill:.2f} ml Part B per liter")
    print(f"  Ratio A:B = {optimal_ml_a_refill:.2f}:{optimal_ml_b_refill:.2f}")
    print(f"  Provides N {optimal_n_refill:.1f} ppm, P {optimal_p_refill:.1f} ppm, K {optimal_k_refill:.1f} ppm")
    print(f"  Ca {optimal_ca_refill:.1f} ppm, Mg {optimal_mg_refill:.1f} ppm, Fe {optimal_fe_refill:.1f} ppm")
    print(f"  - This constant mix maintains concentrations close to targets over {SIM_DAYS} days.")
    print("Note: Estimates for Thai peppers at 85F, 70% RH; uptake depends on concentration. Monitor EC (1.5-2.5 mS/cm), pH (5.8-6.2).")

    # Analytical mean trajectory and heuristic variance
    # Build mean params for simulate
    mean_params = {
        'humidity_scale': HUMIDITY_SCALE,
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    n_mean, p_mean, k_mean, ca_mean, mg_mean, fe_mean, n_uptake_total, cum_mass_mean = simulate(
        ml_a_init, ml_b_init, optimal_ml_a_refill, optimal_ml_b_refill, days=SIM_DAYS, sample_params=mean_params, light_dli=light_dli, air_temp_f=air_temp, humidity=rh
    )
    # Analytic std: include Vmax and Km contributions with floors and climate scaling
    # Climate relative std (dimensionless) consistent with objective
    light_var = (LIGHT_DLI_STD / max(1e-6, LIGHT_DLI_BASE))**2
    temp_var = (AIR_TEMPERATURE_STD / 18.0 * np.log(TEMP_Q10))**2
    humidity_var = HUMIDITY_SCALE_STD**2
    climate_std_rel = np.sqrt(light_var + temp_var + humidity_var)

    REL_STD_FLOOR = {'N': 0.08, 'P': 0.10, 'K': 0.08, 'Ca': 0.10, 'Mg': 0.12, 'Fe': 0.25}
    ABS_STD_FLOOR = {'N': 2.0, 'P': 1.0, 'K': 3.0, 'Ca': 2.0, 'Mg': 1.0, 'Fe': 0.2}

    def approx_std_full(conc_series, v_max_std, v_max_mean, k_m_std, k_m_mean, key):
        stage = np.linspace(0.5, 1.0, len(conc_series))
        rel_var = (v_max_std / max(1e-6, v_max_mean))**2 + (k_m_std / max(1e-6, k_m_mean + np.mean(conc_series)))**2
        base = conc_series * np.sqrt(rel_var) * stage
        base *= (1.0 + climate_std_rel)
        rel_floor = REL_STD_FLOOR[key]
        abs_floor = ABS_STD_FLOOR[key]
        base = np.maximum(base, rel_floor * conc_series)
        base = np.maximum(base, abs_floor)
        return base

    n_std = approx_std_full(n_mean, N_V_MAX_STD, N_V_MAX, N_K_M_STD, N_K_M, 'N')
    p_std = approx_std_full(p_mean, P_V_MAX_STD, P_V_MAX, P_K_M_STD, P_K_M, 'P')
    k_std = approx_std_full(k_mean, K_V_MAX_STD, K_V_MAX, K_K_M_STD, K_K_M, 'K')
    ca_std = approx_std_full(ca_mean, CA_V_MAX_STD, CA_V_MAX, CA_K_M_STD, CA_K_M, 'Ca')
    mg_std = approx_std_full(mg_mean, MG_V_MAX_STD, MG_V_MAX, MG_K_M_STD, MG_K_M, 'Mg')
    fe_std = approx_std_full(fe_mean, FE_V_MAX_STD, FE_V_MAX, FE_K_M_STD, FE_K_M, 'Fe')

    # Water usage approximation with light scaling
    light_factor = max(0.2, light_dli / LIGHT_DLI_BASE)
    daily_w = np.array([daily_water(d, HUMIDITY_SCALE, light_factor, air_temp) for d in range(SIM_DAYS)])
    cum_water_mean = np.cumsum(daily_w)
    cum_water_std = 0.1 * cum_water_mean  # simple proportional uncertainty

    # Calculate cumulative water usage and plant mass (mean +/- std)
    days = np.arange(SIM_DAYS + 1)
    print(f"\nCumulative refill water usage over {SIM_DAYS} days: {cum_water_mean[-1]:.1f} ± {cum_water_std[-1]:.1f} L for 2 plants")
    print(f"Cumulative plant mass (dry weight) over {SIM_DAYS} days: {cum_mass_mean[-1]:.1f} g per plant")

    if not args.no_plot:
        # Plot simulation with error bar shading (mean +/- 1 std)
        target_n = np.array([target_n_ppm(d) for d in range(SIM_DAYS + 1)])
        target_p = np.array([target_p_ppm(d) for d in range(SIM_DAYS + 1)])
        target_k = np.array([target_k_ppm(d) for d in range(SIM_DAYS + 1)])
        target_ca = np.array([target_ca_ppm(d) for d in range(SIM_DAYS + 1)])
        target_mg = np.array([target_mg_ppm(d) for d in range(SIM_DAYS + 1)])
        target_fe = np.array([target_fe_ppm(d) for d in range(SIM_DAYS + 1)])

        # Sensitivity shading around targets (narrower bands for higher sensitivity)
        SENS = OPT_WEIGHTS
        BASE_TOL_REL = {'N': 0.10, 'P': 0.12, 'K': 0.10, 'Ca': 0.10, 'Mg': 0.12, 'Fe': 0.30}
        def sens_band(target, key):
            return target * (BASE_TOL_REL[key] / np.sqrt(max(1e-6, SENS[key]))) * (1.0 + climate_std_rel)
        n_band = sens_band(target_n, 'N')
        p_band = sens_band(target_p, 'P')
        k_band = sens_band(target_k, 'K')
        ca_band = sens_band(target_ca, 'Ca')
        mg_band = sens_band(target_mg, 'Mg')
        fe_band = sens_band(target_fe, 'Fe')

        fig, axs = plt.subplots(7, 1, figsize=(10, 24))
        axs[0].fill_between(days, target_n - n_band, target_n + n_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[0].plot(days, target_n, label='Target N PPM', linestyle='--', color='gray')
        axs[0].plot(days, n_mean, label='Mean Simulated N PPM')
        axs[0].fill_between(days, n_mean - n_std, n_mean + n_std, alpha=0.3, label='±1 Std (analytic)')
        axs[0].set_ylabel('N (ppm)')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].fill_between(days, target_p - p_band, target_p + p_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[1].plot(days, target_p, label='Target P PPM', linestyle='--', color='gray')
        axs[1].plot(days, p_mean, label='Mean Simulated P PPM')
        axs[1].fill_between(days, p_mean - p_std, p_mean + p_std, alpha=0.3, label='±1 Std (analytic)')
        axs[1].set_ylabel('P (ppm)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        axs[2].fill_between(days, target_k - k_band, target_k + k_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[2].plot(days, target_k, label='Target K PPM', linestyle='--', color='gray')
        axs[2].plot(days, k_mean, label='Mean Simulated K PPM')
        axs[2].fill_between(days, k_mean - k_std, k_mean + k_std, alpha=0.3, label='±1 Std (analytic)')
        axs[2].set_ylabel('K (ppm)')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        axs[3].fill_between(days, target_ca - ca_band, target_ca + ca_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[3].plot(days, target_ca, label='Target Ca PPM', linestyle='--', color='gray')
        axs[3].plot(days, ca_mean, label='Mean Simulated Ca PPM')
        axs[3].fill_between(days, ca_mean - ca_std, ca_mean + ca_std, alpha=0.3, label='±1 Std (analytic)')
        axs[3].set_ylabel('Ca (ppm)')
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)

        axs[4].fill_between(days, target_mg - mg_band, target_mg + mg_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[4].plot(days, target_mg, label='Target Mg PPM', linestyle='--', color='gray')
        axs[4].plot(days, mg_mean, label='Mean Simulated Mg PPM')
        axs[4].fill_between(days, mg_mean - mg_std, mg_mean + mg_std, alpha=0.3, label='±1 Std (analytic)')
        axs[4].set_ylabel('Mg (ppm)')
        axs[4].legend()
        axs[4].grid(True, alpha=0.3)

        axs[5].fill_between(days, target_fe - fe_band, target_fe + fe_band, color='orange', alpha=0.15, label='Sensitivity band')
        axs[5].plot(days, target_fe, label='Target Fe PPM', linestyle='--', color='gray')
        axs[5].plot(days, fe_mean, label='Mean Simulated Fe PPM')
        axs[5].fill_between(days, fe_mean - fe_std, fe_mean + fe_std, alpha=0.3, label='±1 Std (analytic)')
        axs[5].set_ylabel('Fe (ppm)')
        axs[5].legend()
        axs[5].grid(True, alpha=0.3)

        axs[6].plot(days[1:], cum_water_mean, label='Mean Cumulative Water Use (L for 2 plants)', color='purple')
        axs[6].fill_between(days[1:], cum_water_mean - cum_water_std, cum_water_mean + cum_water_std, alpha=0.3, color='purple')
        axs[6].plot(days, cum_mass_mean, label='Mean Cumulative Plant Mass (g dry weight, per plant)', color='green')
        axs[6].set_xlabel('Days')
        axs[6].set_ylabel('Water (L) / Mass (g)')
        axs[6].legend()
        axs[6].grid(True, alpha=0.3)

        plt.tight_layout()
        if args.output:
            plt.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {args.output}")
        else:
            plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Hydroponic nutrient optimization simulator')
    parser.add_argument('--days', type=int, default=SIM_DAYS,
                       help=f'Simulation duration in days (default: {SIM_DAYS})')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting and only show optimization results')
    parser.add_argument('--output', type=str, default=None,
                       help='Save plot to file (e.g., hydroponic_results.png)')
    parser.add_argument('--light-dli', type=float, default=LIGHT_DLI,
                       help=f'Daily light integral (mol/m^2/day), default {LIGHT_DLI}')
    parser.add_argument('--risk-lambda', type=float, default=RISK_LAMBDA,
                       help=f'Variance penalty in probabilistic health, default {RISK_LAMBDA}')
    parser.add_argument('--air-temp-f', type=float, default=AIR_TEMPERATURE,
                       help=f'Air temperature in Fahrenheit (default: {AIR_TEMPERATURE})')
    parser.add_argument('--humidity', type=float, default=HUMIDITY,
                       help=f'Relative humidity fraction 0-1 (default: {HUMIDITY})')
    return parser.parse_args()

if __name__ == '__main__':
    main()
