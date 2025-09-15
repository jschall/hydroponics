import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse

# Constants
RESERVOIR_VOLUME = 4.0  # liters
AIR_TEMPERATURE = 85.0  # Fahrenheit
HUMIDITY = 0.7  # 70% relative humidity
HUMIDITY_STD = 0.07  # Absolute RH fraction std (e.g., 0.05 = 5% RH)
LIGHT_DLI = 25.0  # default DLI
LIGHT_DLI_STD = 5.0  # std for DLI
AIR_TEMPERATURE_STD = 5.0  # Fahrenheit std
TEMP_Q10 = 2.0  # Q10 temperature factor for uptake scaling
RISK_LAMBDA = 1.0  # weight on variance in expected loss

# Phenology: germination lag and ramp for growth-stage-controlled uptake
STAGE_S_MIN = 0.02  # logistic lower asymptote
STAGE_D50 = 55.0  # day at 50% stage
STAGE_K = 0.15  # logistic slope

# Light and ET parameters
K_LIGHT = 15.0  # mol/m^2/day for half-saturation of light response
BETA_VPD = 1.0  # exponent for VPD effect on ET
WATER_C_L_PER_PLANT = 0.4  # scaling constant for ET liters per plant per day at stage=1

# Target tolerance (Gaussian) parameters used by objective and plotting bands
TGT_REL = {'N': 0.10, 'P': 0.12, 'K': 0.10, 'Ca': 0.10, 'Mg': 0.12, 'Fe': 0.30}
TGT_ABS = {'N': 5.0,  'P': 2.0,  'K': 10.0, 'Ca': 5.0,  'Mg': 2.0,  'Fe': 0.2}
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

def logistic_stage(day, s_min=STAGE_S_MIN, d50=STAGE_D50, k=STAGE_K):
    return s_min + (1.0 - s_min) / (1.0 + np.exp(-k * (day - d50)))

def vpd_kpa(temp_f, rh_frac):
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    return es * max(0.0, (1.0 - rh_frac))

def light_saturation(dli, k_light=K_LIGHT):
    return dli / (dli + k_light)

# Define daily water uptake for 2 Thai pepper plants (L/day total)
def daily_water(day, light_dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    s = logistic_stage(day)
    q10_factor = TEMP_Q10 ** ((temp_f - 77.0) / 18.0)
    vpd_term = max(0.05, vpd_kpa(temp_f, rh_frac)) ** BETA_VPD
    light_term = light_saturation(light_dli)
    w_per_plant = WATER_C_L_PER_PLANT * s * q10_factor * light_term * vpd_term
    return 2.0 * w_per_plant

# Humidity scale mapping from climate via VPD (kPa)
def humidity_scale_from_climate(temp_f, rh_frac, alpha=0.6):
    # Backward-compat shim; daily_water now uses VPD directly. Keep this for uncertainty mapping if needed.
    return float(np.clip((vpd_kpa(temp_f, rh_frac) / max(1e-6, vpd_kpa(AIR_TEMPERATURE, HUMIDITY))) ** alpha, 0.2, 1.8))

# Concentration-dependent uptake function (mg/day/2 plants)
def uptake(c, d, v_max, k_m, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    q10_factor = TEMP_Q10 ** ((temp_f - 77.0) / 18.0)
    stage_factor = logistic_stage(d)
    light_factor = light_saturation(dli)
    vpd_factor = max(0.3, vpd_kpa(temp_f, rh_frac)) ** 0.2  # mild modulation on uptake
    v_max_adj = v_max * stage_factor * light_factor * q10_factor * vpd_factor
    return v_max_adj * c / (k_m + c)

# Specific uptake functions for each nutrient
def uptake_n(c, d, v_max=N_V_MAX, k_m=N_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

def uptake_p(c, d, v_max=P_V_MAX, k_m=P_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

def uptake_k(c, d, v_max=K_V_MAX, k_m=K_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

def uptake_ca(c, d, v_max=CA_V_MAX, k_m=CA_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

def uptake_mg(c, d, v_max=MG_V_MAX, k_m=MG_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

def uptake_fe(c, d, v_max=FE_V_MAX, k_m=FE_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac)

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

    # No precomputed factors; use functions directly

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
        W = daily_water(d, light_dli, air_temp_f, humidity)

        # Concentration-dependent uptake with sampled params
        U_n = uptake_n(n_ppm[d], d, n_v_max, n_k_m, days, light_dli, air_temp_f, humidity)
        U_p = uptake_p(p_ppm[d], d, p_v_max, p_k_m, days, light_dli, air_temp_f, humidity)
        U_k = uptake_k(k_ppm[d], d, k_v_max, k_k_m, days, light_dli, air_temp_f, humidity)
        U_ca = uptake_ca(ca_ppm[d], d, ca_v_max, ca_k_m, days, light_dli, air_temp_f, humidity)
        U_mg = uptake_mg(mg_ppm[d], d, mg_v_max, mg_k_m, days, light_dli, air_temp_f, humidity)
        U_fe = uptake_fe(fe_ppm[d], d, fe_v_max, fe_k_m, days, light_dli, air_temp_f, humidity)

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

# Centralized computation of means and variances (±σ propagation of constant climate and uptake parameters)
def compute_means_and_variances(ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days, sample_params, light_dli, air_temp_f, humidity):
    if sample_params is None:
        sample_params = {}

    # Build mean parameter set
    mean_params = {
        'n_v_max': sample_params.get('n_v_max', N_V_MAX),
        'p_v_max': sample_params.get('p_v_max', P_V_MAX),
        'k_v_max': sample_params.get('k_v_max', K_V_MAX),
        'ca_v_max': sample_params.get('ca_v_max', CA_V_MAX),
        'mg_v_max': sample_params.get('mg_v_max', MG_V_MAX),
        'fe_v_max': sample_params.get('fe_v_max', FE_V_MAX),
        'n_k_m': sample_params.get('n_k_m', N_K_M),
        'p_k_m': sample_params.get('p_k_m', P_K_M),
        'k_k_m': sample_params.get('k_k_m', K_K_M),
        'ca_k_m': sample_params.get('ca_k_m', CA_K_M),
        'mg_k_m': sample_params.get('mg_k_m', MG_K_M),
        'fe_k_m': sample_params.get('fe_k_m', FE_K_M),
        'n_to_biomass': sample_params.get('n_to_biomass', N_TO_BIOMASS),
    }

    n_mean, p_mean, k_mean, ca_mean, mg_mean, fe_mean, n_uptake_total, cum_mass_mean = simulate(
        ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days=days, sample_params=mean_params, light_dli=light_dli, air_temp_f=air_temp_f, humidity=humidity
    )

    var_n = np.zeros(days + 1)
    var_p = np.zeros(days + 1)
    var_k = np.zeros(days + 1)
    var_ca = np.zeros(days + 1)
    var_mg = np.zeros(days + 1)
    var_fe = np.zeros(days + 1)

    def add_var_for_params(params_plus, params_minus, ldli, tempf, rh):
        nonlocal var_n, var_p, var_k, var_ca, var_mg, var_fe
        n_p, p_p, k_p, ca_p, mg_p, fe_p, _, _ = simulate(ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days=days, sample_params=params_plus, light_dli=ldli, air_temp_f=tempf, humidity=rh)
        n_m, p_m, k_m_, ca_m, mg_m, fe_m, _, _ = simulate(ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, days=days, sample_params=params_minus, light_dli=ldli, air_temp_f=tempf, humidity=rh)
        var_n += ((n_p - n_m) / 2.0) ** 2
        var_p += ((p_p - p_m) / 2.0) ** 2
        var_k += ((k_p - k_m_) / 2.0) ** 2
        var_ca += ((ca_p - ca_m) / 2.0) ** 2
        var_mg += ((mg_p - mg_m) / 2.0) ** 2
        var_fe += ((fe_p - fe_m) / 2.0) ** 2

    # Uptake parameter stds
    std_specs = [
        ('n_v_max', N_V_MAX_STD), ('p_v_max', P_V_MAX_STD), ('k_v_max', K_V_MAX_STD), ('ca_v_max', CA_V_MAX_STD), ('mg_v_max', MG_V_MAX_STD), ('fe_v_max', FE_V_MAX_STD),
        ('n_k_m', N_K_M_STD), ('p_k_m', P_K_M_STD), ('k_k_m', K_K_M_STD), ('ca_k_m', CA_K_M_STD), ('mg_k_m', MG_K_M_STD), ('fe_k_m', FE_K_M_STD),
    ]
    for name, std in std_specs:
        p_plus = dict(mean_params); p_minus = dict(mean_params)
        p_plus[name] = p_plus[name] + std
        p_minus[name] = p_minus[name] - std
        add_var_for_params(p_plus, p_minus, light_dli, air_temp_f, humidity)

    # Climate parameters (constant across days)
    add_var_for_params(mean_params, mean_params, light_dli + LIGHT_DLI_STD, air_temp_f, humidity)
    add_var_for_params(mean_params, mean_params, max(0.0, light_dli - LIGHT_DLI_STD), air_temp_f, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f + AIR_TEMPERATURE_STD, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f - AIR_TEMPERATURE_STD, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f, min(0.99, humidity + HUMIDITY_STD))
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f, max(0.0, humidity - HUMIDITY_STD))

    # Water: daily and cumulative
    daily_w_mean = np.array([daily_water(d, light_dli, air_temp_f, humidity) for d in range(days)])
    cum_water_mean = np.cumsum(daily_w_mean)

    def halfdiff_sq(a, b):
        return ((np.array(a) - np.array(b)) / 2.0) ** 2

    daily_w_dli_plus = [daily_water(d, light_dli + LIGHT_DLI_STD, air_temp_f, humidity) for d in range(days)]
    daily_w_dli_minus = [daily_water(d, max(0.0, light_dli - LIGHT_DLI_STD), air_temp_f, humidity) for d in range(days)]
    daily_w_temp_plus = [daily_water(d, light_dli, air_temp_f + AIR_TEMPERATURE_STD, humidity) for d in range(days)]
    daily_w_temp_minus = [daily_water(d, light_dli, air_temp_f - AIR_TEMPERATURE_STD, humidity) for d in range(days)]
    daily_w_rh_plus = [daily_water(d, light_dli, air_temp_f, min(0.99, humidity + HUMIDITY_STD)) for d in range(days)]
    daily_w_rh_minus = [daily_water(d, light_dli, air_temp_f, max(0.0, humidity - HUMIDITY_STD)) for d in range(days)]

    daily_w_var = halfdiff_sq(daily_w_dli_plus, daily_w_dli_minus) + \
                   halfdiff_sq(daily_w_temp_plus, daily_w_temp_minus) + \
                   halfdiff_sq(daily_w_rh_plus, daily_w_rh_minus)
    # Constant-in-time parameter uncertainties induce fully time-correlated effects per parameter.
    # Compute cumulative variance from cumulative trajectories under ±σ per parameter and sum contributions.
    C_dli_plus = np.cumsum(daily_w_dli_plus)
    C_dli_minus = np.cumsum(daily_w_dli_minus)
    C_temp_plus = np.cumsum(daily_w_temp_plus)
    C_temp_minus = np.cumsum(daily_w_temp_minus)
    C_rh_plus = np.cumsum(daily_w_rh_plus)
    C_rh_minus = np.cumsum(daily_w_rh_minus)
    cum_water_var = halfdiff_sq(C_dli_plus, C_dli_minus) + \
                    halfdiff_sq(C_temp_plus, C_temp_minus) + \
                    halfdiff_sq(C_rh_plus, C_rh_minus)

    return {
        'mean': {'N': n_mean, 'P': p_mean, 'K': k_mean, 'Ca': ca_mean, 'Mg': mg_mean, 'Fe': fe_mean},
        'var': {'N': var_n, 'P': var_p, 'K': var_k, 'Ca': var_ca, 'Mg': var_mg, 'Fe': var_fe},
        'n_uptake_total': n_uptake_total,
        'cum_mass_mean': cum_mass_mean,
        'daily_water_mean': daily_w_mean,
        'daily_water_var': daily_w_var,
        'cum_water_mean': cum_water_mean,
        'cum_water_var': cum_water_var,
    }

# Joint optimization bounds for [ml_a_init, ml_b_init, ml_a_refill, ml_b_refill]
bounds = [(0, 10), (0, 10), (0, 10), (0, 10)]

# Objective for refill: probabilistic health with analytical uncertainty
def objective_refill(x, sim_days=SIM_DAYS, light_dli=LIGHT_DLI, air_temp_f=AIR_TEMPERATURE, humidity=HUMIDITY):
    if len(x) == 2:
        ml_a_init, ml_b_init = 1.0, 1.0  # use default initial concentrations if only refill provided
        ml_a_refill, ml_b_refill = x
    else:
        ml_a_init, ml_b_init, ml_a_refill, ml_b_refill = x
    target_n = np.array([target_n_ppm(d) for d in range(sim_days + 1)])
    target_p = np.array([target_p_ppm(d) for d in range(sim_days + 1)])
    target_k = np.array([target_k_ppm(d) for d in range(sim_days + 1)])
    target_ca = np.array([target_ca_ppm(d) for d in range(sim_days + 1)])
    target_mg = np.array([target_mg_ppm(d) for d in range(sim_days + 1)])
    target_fe = np.array([target_fe_ppm(d) for d in range(sim_days + 1)])
    # Use centralized means/variances
    mean_params = {
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    results = compute_means_and_variances(ml_a_init, ml_b_init, ml_a_refill, ml_b_refill, sim_days, mean_params, light_dli, air_temp_f, humidity)
    n_sim = results['mean']['N']; p_sim = results['mean']['P']; k_sim = results['mean']['K']
    ca_sim = results['mean']['Ca']; mg_sim = results['mean']['Mg']; fe_sim = results['mean']['Fe']
    var_n = results['var']['N']; var_p = results['var']['P']; var_k = results['var']['K']
    var_ca = results['var']['Ca']; var_mg = results['var']['Mg']; var_fe = results['var']['Fe']

    # Gaussian target loss: per-day NLL with combined variance (model + target)
    def target_var(arr_target, key):
        std = np.maximum(TGT_REL[key] * arr_target, TGT_ABS[key])
        return std**2

    n_var_tgt = target_var(target_n, 'N')
    p_var_tgt = target_var(target_p, 'P')
    k_var_tgt = target_var(target_k, 'K')
    ca_var_tgt = target_var(target_ca, 'Ca')
    mg_var_tgt = target_var(target_mg, 'Mg')
    fe_var_tgt = target_var(target_fe, 'Fe')

    eps = 1e-9
    n_tot_var = var_n + n_var_tgt + eps
    p_tot_var = var_p + p_var_tgt + eps
    k_tot_var = var_k + k_var_tgt + eps
    ca_tot_var = var_ca + ca_var_tgt + eps
    mg_tot_var = var_mg + mg_var_tgt + eps
    fe_tot_var = var_fe + fe_var_tgt + eps

    loss_n = np.mean(((n_sim - target_n)**2) / n_tot_var + np.log(n_tot_var))
    loss_p = np.mean(((p_sim - target_p)**2) / p_tot_var + np.log(p_tot_var))
    loss_k = np.mean(((k_sim - target_k)**2) / k_tot_var + np.log(k_tot_var))
    loss_ca = np.mean(((ca_sim - target_ca)**2) / ca_tot_var + np.log(ca_tot_var))
    loss_mg = np.mean(((mg_sim - target_mg)**2) / mg_tot_var + np.log(mg_tot_var))
    loss_fe = np.mean(((fe_sim - target_fe)**2) / fe_tot_var + np.log(fe_tot_var))

    return loss_n + loss_p + loss_k + loss_ca + loss_mg + loss_fe

def main():
    args = parse_args()

    # Override global constants with CLI args
    global SIM_DAYS
    SIM_DAYS = args.days
    light_dli = args.light_dli
    air_temp = args.air_temp_f
    rh = args.humidity
    # Update tunables
    global K_LIGHT, STAGE_D50, STAGE_K
    K_LIGHT = args.k_light
    STAGE_D50 = args.stage_d50
    STAGE_K = args.stage_k

# Optimize refill concentrations
    print("Optimizing refill concentrations for Parts A and B...")
    # Update global risk lambda from CLI
    global RISK_LAMBDA
    RISK_LAMBDA = args.risk_lambda

    # Jointly optimize [ml_a_init, ml_b_init, ml_a_refill, ml_b_refill]
    result = minimize(lambda x: objective_refill(x, SIM_DAYS, light_dli, air_temp, rh), [1.0,1.0,1.0,1.0], bounds=bounds, method='SLSQP')
    ml_a_init, ml_b_init, optimal_ml_a_refill, optimal_ml_b_refill = result.x
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

    # Centralized means and variances for plotting
    mean_params = {
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    results = compute_means_and_variances(ml_a_init, ml_b_init, optimal_ml_a_refill, optimal_ml_b_refill, SIM_DAYS, mean_params, light_dli, air_temp, rh)
    n_mean = results['mean']['N']; p_mean = results['mean']['P']; k_mean = results['mean']['K']
    ca_mean = results['mean']['Ca']; mg_mean = results['mean']['Mg']; fe_mean = results['mean']['Fe']
    n_var_s = results['var']['N']; p_var_s = results['var']['P']; k_var_s = results['var']['K']
    ca_var_s = results['var']['Ca']; mg_var_s = results['var']['Mg']; fe_var_s = results['var']['Fe']
    n_std = np.sqrt(n_var_s); p_std = np.sqrt(p_var_s); k_std = np.sqrt(k_var_s)
    ca_std = np.sqrt(ca_var_s); mg_std = np.sqrt(mg_var_s); fe_std = np.sqrt(fe_var_s)
    # Floors
    REL_STD_FLOOR = {'N': 0.08, 'P': 0.10, 'K': 0.08, 'Ca': 0.10, 'Mg': 0.12, 'Fe': 0.25}
    ABS_STD_FLOOR = {'N': 2.0, 'P': 1.0, 'K': 3.0, 'Ca': 2.0, 'Mg': 1.0, 'Fe': 0.2}
    n_std = np.maximum(n_std, np.maximum(REL_STD_FLOOR['N'] * n_mean, ABS_STD_FLOOR['N']))
    p_std = np.maximum(p_std, np.maximum(REL_STD_FLOOR['P'] * p_mean, ABS_STD_FLOOR['P']))
    k_std = np.maximum(k_std, np.maximum(REL_STD_FLOOR['K'] * k_mean, ABS_STD_FLOOR['K']))
    ca_std = np.maximum(ca_std, np.maximum(REL_STD_FLOOR['Ca'] * ca_mean, ABS_STD_FLOOR['Ca']))
    mg_std = np.maximum(mg_std, np.maximum(REL_STD_FLOOR['Mg'] * mg_mean, ABS_STD_FLOOR['Mg']))
    fe_std = np.maximum(fe_std, np.maximum(REL_STD_FLOOR['Fe'] * fe_mean, ABS_STD_FLOOR['Fe']))

    daily_w = results['daily_water_mean']
    daily_w_std = np.sqrt(results['daily_water_var'])
    cum_water_mean = results['cum_water_mean']
    cum_water_std = np.sqrt(results['cum_water_var'])
    n_uptake_total = results['n_uptake_total']
    cum_mass_mean = results['cum_mass_mean']

    # Calculate cumulative water usage and plant mass (mean +/- std)
    days = np.arange(SIM_DAYS + 1)
    print(f"\nCumulative refill water usage over {SIM_DAYS} days: {cum_water_mean[-1]:.1f} ± {cum_water_std[-1]:.1f} L for 2 plants")
    # Biomass uncertainty via delta method: var(mass) ≈ (N_TO_BIOMASS_STD * uptake)^2 + (N_TO_BIOMASS * uptake_std)^2
    # Approximate uptake_std from nutrient concentration variance proxy (N) using centralized results
    uptake_std = np.sqrt(np.maximum(0.0, results['var']['N'][-1]))
    mass_std = np.sqrt((N_TO_BIOMASS_STD * n_uptake_total[-1])**2 + (N_TO_BIOMASS * uptake_std)**2) / 2.0
    print(f"Cumulative plant mass (dry weight) over {SIM_DAYS} days: {cum_mass_mean[-1]:.1f} ± {mass_std:.1f} g per plant")

    if not args.no_plot:
        # Plot simulation with error bar shading (mean +/- 1 std)
        target_n = np.array([target_n_ppm(d) for d in range(SIM_DAYS + 1)])
        target_p = np.array([target_p_ppm(d) for d in range(SIM_DAYS + 1)])
        target_k = np.array([target_k_ppm(d) for d in range(SIM_DAYS + 1)])
        target_ca = np.array([target_ca_ppm(d) for d in range(SIM_DAYS + 1)])
        target_mg = np.array([target_mg_ppm(d) for d in range(SIM_DAYS + 1)])
        target_fe = np.array([target_fe_ppm(d) for d in range(SIM_DAYS + 1)])

        # Plot target Gaussian bands using same TGT_REL/TGT_ABS as objective
        def tgt_std(arr_target, key):
            return np.maximum(TGT_REL[key] * arr_target, TGT_ABS[key])
        n_band = tgt_std(target_n, 'N')
        p_band = tgt_std(target_p, 'P')
        k_band = tgt_std(target_k, 'K')
        ca_band = tgt_std(target_ca, 'Ca')
        mg_band = tgt_std(target_mg, 'Mg')
        fe_band = tgt_std(target_fe, 'Fe')

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
        axs[6].set_ylabel('Cumulative Water (L) / Mass (g)')
        axs[6].legend(loc='upper left')
        axs[6].grid(True, alpha=0.3)

        # Add daily water use on a separate axis
        ax6_twin = axs[6].twinx()
        ax6_twin.plot(days[:SIM_DAYS], daily_w, label='Daily Water Use (L for 2 plants)', color='blue', linestyle='--', alpha=0.8)
        ax6_twin.fill_between(days[:SIM_DAYS], daily_w - daily_w_std, daily_w + daily_w_std, alpha=0.3, color='blue')
        ax6_twin.set_ylabel('Daily Water Use (L)', color='blue')
        ax6_twin.tick_params(axis='y', labelcolor='blue')

        # Combine legends from both axes
        lines1, labels1 = axs[6].get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        axs[6].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

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
    parser.add_argument('--k-light', type=float, default=K_LIGHT,
                       help=f'Half-saturation DLI for light response (default: {K_LIGHT})')
    parser.add_argument('--stage-d50', type=float, default=STAGE_D50,
                       help=f'Day at 50% growth stage (default: {STAGE_D50})')
    parser.add_argument('--stage-k', type=float, default=STAGE_K,
                       help=f'Logistic stage slope (default: {STAGE_K})')
    return parser.parse_args()

if __name__ == '__main__':
    main()
