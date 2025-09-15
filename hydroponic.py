import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse

# Constants
RESERVOIR_VOLUME = 2.0  # liters
AIR_TEMPERATURE = 82.0  # Fahrenheit, average for South FL October garage
HUMIDITY = 0.75  # 75% relative humidity
HUMIDITY_STD = 0.1  # Increased for garage variability
LIGHT_DLI = 20.0  # Adjusted for AeroGarden LED setup
LIGHT_DLI_STD = 4.0  # Reduced but still accounts for some natural light variation
AIR_TEMPERATURE_STD = 6.0  # Increased for daily/seasonal swings
TEMP_Q10 = 2.0  # Q10 temperature factor for uptake scaling
RISK_LAMBDA = 1.0  # weight on variance in expected loss

# Tap water background minerals (South Florida / Pembroke Pines)
# TAP_CA_MG_L = 80.0   # Calcium in tap water (mg/L) - higher due to limestone geology
# TAP_MG_MG_L = 20.0   # Magnesium in tap water (mg/L) - moderate hardness
# TAP_NA_MG_L = 25.0   # Sodium in tap water (mg/L)
# TAP_K_MG_L = 1.0     # Potassium in tap water (mg/L) - very low
# TAP_ALK_MG_L = 140.0 # Alkalinity in tap water (mg/L CaCO3) - moderate to high
# TAP_PH = 7.8         # pH of tap water - slightly alkaline from limestone buffering

# RO water baseline (post-RO, low mineral)
TAP_CA_MG_L = 0.0    # Calcium in RO water (mg/L)
TAP_MG_MG_L = 0.0    # Magnesium in RO water (mg/L)
TAP_NA_MG_L = 1.0    # Sodium in RO water (mg/L)
TAP_K_MG_L = 0.0     # Potassium in RO water (mg/L)
TAP_ALK_MG_L = 10.0  # Alkalinity in RO water (mg/L CaCO3)
TAP_PH = 6.0         # pH reference for RO (weakly buffered; align with plant target)

# Phenology: germination lag and ramp for growth-stage-controlled uptake
STAGE_S_MIN = 0.02  # logistic lower asymptote
STAGE_D50 = 45.0  # Adjusted for faster Thai chili maturation
STAGE_K = 0.2  # Steeper logistic for quicker ramp-up

# Light and ET parameters
K_LIGHT = 15.0  # mol/m^2/day for half-saturation of light response
BETA_VPD = 1.0  # exponent for VPD effect on ET
WATER_C_L_PER_PLANT = 0.3  # Slightly reduced for smaller plants/AeroGarden setup
SLA_M2_PER_G = 0.02  # specific leaf area (m^2 leaf per g dry biomass)
LAI_MAX = 3.0
PLANT_SPACING_M2 = 0.10  # ground area per plant
RESERVOIR_AREA_M2 = 0.05

# Penman–Monteith-lite parameters
WIND_SPEED = 0.3  # m/s indoor
PSYCHROMETRIC_GAMMA = 0.066  # kPa/°C
RAD_MJ_PER_MOL_PAR = 0.218  # MJ/m^2 per mol PAR
NET_RAD_MULT = 1.6  # include NIR
KC_MIN = 0.6
KC_MAX = 1.15
FE_DECAY_RATE = 0.001  # per-day fractional Fe loss (shaded reservoir; low photodegradation)
FE_DECAY_RATE_STD = 0.0005
PH_BASE = 6.0
PH_FE_SENS = 1.0  # Fe K_m sensitivity per pH unit above 6.5
K_PH_DRIFT = 0.0002
K_PH_RECOVER = 0.02
PH_STD = 0.2
ALK_INIT_MG_L = 50.0
K_ALK_CONV = 0.5  # mg CaCO3 per mg acid-equivalent consumed (increased for visibility)
ALK_BUFFER_BETA = 0.01  # pH drift damping per mg/L alkalinity
CORR_T_RH = -0.5
CORR_T_DLI = 0.2
CORR_DLI_RH = -0.2
FE_REFILL_DECAY_RATE = 0.002  # per-day decay in prepared refill mix (low light exposure)
EVAP_COEFF_MM_PER_KPA = 0.5  # Tuned for completely covered reservoir (very low evap)
CI_Z = 1.96  # 95% CI multiplier for Gaussian

# Generalized chemical system
# Each chemical contributes elemental ppm per ml/L of concentrate added, and may change alkalinity (mg/L CaCO3) per ml/L (negative for acids)
# Users can edit this list to match their available products (3-part, 6-part, etc.)
# CHEMICALS = [
#     { 'name': 'Part A', 'per_ml_ppm': {'N': 60.0, 'P': 0.0, 'K': 41.5, 'Ca': 50.0, 'Mg': 0.0, 'Fe': 1.2}, 'alk_change_mg_per_ml': 0.0, 'ml_bounds': (0.0, 10.0) },
#     { 'name': 'Part B', 'per_ml_ppm': {'N': 10.0, 'P': 21.8, 'K': 49.8, 'Ca': 0.0, 'Mg': 12.0, 'Fe': 0.0}, 'alk_change_mg_per_ml': 0.0, 'ml_bounds': (0.0, 10.0) },
# ]

# Green Planet GP3 system (ppm per ml/L). Conversions: P2O5->P factor 0.436, K2O->K factor 0.83; elemental % uses factor 10 ppm per 1%.
CHEMICALS = [
    { 'name': 'Part A', 'per_ml_ppm': {'N': 60.0, 'P': 0.0, 'K': 41.5, 'Ca': 50.0, 'Mg': 0.0, 'Fe': 1.2}, 'alk_change_mg_per_ml': 0.0, 'ml_bounds': (0.0, 10.0) },
    { 'name': 'Part B', 'per_ml_ppm': {'N': 10.0, 'P': 21.8, 'K': 49.8, 'Ca': 0.0, 'Mg': 12.0, 'Fe': 0.0}, 'alk_change_mg_per_ml': 0.0, 'ml_bounds': (0.0, 10.0) },
    # {
    #     'name': 'GP3 Grow',
    #     'per_ml_ppm': {
    #         'N': 20.0,
    #         'P': 1.0 * 0.436 * 10.0,   # 1% P2O5
    #         'K': 6.0 * 0.83 * 10.0,   # 6% K2O
    #         'Ca': 0.0,
    #         'Mg': 5.0,
    #         'Fe': 0.0,
    #     },
    #     'alk_change_mg_per_ml': 0.0,
    #     'ml_bounds': (0.0, 10.0),
    # },
    # {
    #     'name': 'GP3 Bloom',
    #     'per_ml_ppm': {
    #         'N': 0.0,
    #         'P': 5.0 * 0.436 * 10.0,   # 5% P2O5
    #         'K': 4.0 * 0.83 * 10.0,    # 4% K2O
    #         'Ca': 0.0,
    #         'Mg': 15.0,
    #         'Fe': 0.0,
    #     },
    #     'alk_change_mg_per_ml': 0.0,
    #     'ml_bounds': (0.0, 10.0),
    # },
    # {
    #     'name': 'GP3 Micro',
    #     'per_ml_ppm': {
    #         'N': 50.0,                  # 5% N
    #         'P': 0.0,
    #         'K': 1.0 * 0.83 * 10.0,    # 1% K2O
    #         'Ca': 60.0,                 # 6% Ca
    #         'Mg': 0.0,
    #         'Fe': 1.0,                  # 0.1% Fe chelated
    #     },
    #     'alk_change_mg_per_ml': 0.0,
    #     'ml_bounds': (0.0, 10.0),
    # },
    # Optional nitric acid (edit to your stock concentration or remove if not used)
    {
        'name': 'Nitric Acid',
        'per_ml_ppm': {'N': 150.0},
        'alk_change_mg_per_ml': -400.0,
        'ml_bounds': (0.0, 5.0),
    },
]

def chem_nutrient_ppm_from_ml(mls):
    n = p = k = ca = mg = fe = 0.0
    for m, chem in zip(mls, CHEMICALS):
        comp = chem.get('per_ml_ppm', {})
        n += comp.get('N', 0.0) * m
        p += comp.get('P', 0.0) * m
        k += comp.get('K', 0.0) * m
        ca += comp.get('Ca', 0.0) * m
        mg += comp.get('Mg', 0.0) * m
        fe += comp.get('Fe', 0.0) * m
    return n, p, k, ca, mg, fe

def chem_alk_change_from_ml(mls):
    delta = 0.0
    for m, chem in zip(mls, CHEMICALS):
        delta += chem.get('alk_change_mg_per_ml', 0.0) * m
    return delta

def _split_x(x):
    num_chems = len(CHEMICALS)
    init_ml = list(x[:num_chems])
    refill_ml = list(x[num_chems:2*num_chems])
    alk_init_pt = x[-2]
    alk_refill_pt = x[-1]
    return init_ml, refill_ml, alk_init_pt, alk_refill_pt

# Three-phase stoichiometric schedule (day-based)
PHASE_GERM_END = 14
PHASE_VEG_END = 55
def phase_for_day(day):
    if day <= PHASE_GERM_END:
        return 'germ'
    if day <= PHASE_VEG_END:
        return 'veg'
    return 'fruit'

PHASE_RATIOS = {
    'germ':  {'P': 0.25, 'K': 0.90, 'Ca': 0.35, 'Mg': 0.08, 'Fe': 0.0030},
    'veg':   {'P': 0.20, 'K': 1.50, 'Ca': 0.45, 'Mg': 0.10, 'Fe': 0.0035},
    'fruit': {'P': 0.30, 'K': 2.00, 'Ca': 0.60, 'Mg': 0.12, 'Fe': 0.0040},
}

# Target tolerance (Gaussian) parameters used by objective and plotting bands
TGT_REL = {'N': 0.2, 'P': 0.25, 'K': 0.2, 'Ca': 0.2, 'Mg': 0.2, 'Fe': 0.5}
TGT_ABS = {'N': 4.0,  'P': 1.5,  'K': 8.0, 'Ca': 4.0,  'Mg': 1.5,  'Fe': 0.15}
TOX_MAX = {'N': 300.0, 'P': 100.0, 'K': 300.0, 'Ca': 190.0, 'Mg': 70.0, 'Fe': 5.0}
# Back-compat placeholders (not used by generalized flow but kept to avoid refactoring print math elsewhere).
N_A = CHEMICALS[0]['per_ml_ppm'].get('N', 0.0)
N_B = CHEMICALS[1]['per_ml_ppm'].get('N', 0.0)
P_A = CHEMICALS[0]['per_ml_ppm'].get('P', 0.0)
P_B = CHEMICALS[1]['per_ml_ppm'].get('P', 0.0)
K_A = CHEMICALS[0]['per_ml_ppm'].get('K', 0.0)
K_B = CHEMICALS[1]['per_ml_ppm'].get('K', 0.0)
CA_A = CHEMICALS[0]['per_ml_ppm'].get('Ca', 0.0)
CA_B = CHEMICALS[1]['per_ml_ppm'].get('Ca', 0.0)
MG_A = CHEMICALS[0]['per_ml_ppm'].get('Mg', 0.0)
MG_B = CHEMICALS[1]['per_ml_ppm'].get('Mg', 0.0)
FE_A = CHEMICALS[0]['per_ml_ppm'].get('Fe', 0.0)
FE_B = CHEMICALS[1]['per_ml_ppm'].get('Fe', 0.0)
SIM_DAYS = 90  # Per project goal of 90-day unattended grow
N_TO_BIOMASS = 0.02  # g biomass per mg N uptake (approx., peppers ~2-3% N by dry weight)
N_TO_BIOMASS_STD = 0.005  # Increased variance for biomass conversion uncertainty

# Michaelis-Menten parameters (V_max in mg/day/2 plants, K_m in ppm) with std devs for variance
N_V_MAX = 59.2  # Reduced 20% for smaller Thai plants
N_V_MAX_STD = 14.8  # 25% std for conservatism (59.2 * 0.25 = 14.8)
P_V_MAX = 27.2
P_V_MAX_STD = 6.8
K_V_MAX = 92.8
K_V_MAX_STD = 23.2
CA_V_MAX = 19.52
CA_V_MAX_STD = 4.88
MG_V_MAX = 9.28
MG_V_MAX_STD = 2.32
FE_V_MAX = 0.496
FE_V_MAX_STD = 0.124
N_K_M = 100.0    # half-saturation for N (~50% target 200 ppm)
N_K_M_STD = 25.0  # Increased to 25% for conservatism
P_K_M = 30.0     # half-saturation for P (~50% target 60 ppm)
P_K_M_STD = 7.5
K_K_M = 135.0    # half-saturation for K (~50% target 270 ppm)
K_K_M_STD = 33.75
CA_K_M = 85.0    # half-saturation for Ca (~50% target 170 ppm)
CA_K_M_STD = 21.25
MG_K_M = 28.0    # half-saturation for Mg (~50% target 56 ppm)
MG_K_M_STD = 7.0
FE_K_M = 1.5     # half-saturation for Fe (~50% target 3 ppm)
FE_K_M_STD = 0.375

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
def _lai_from_biomass(biomass_g):
    lai = min(LAI_MAX, SLA_M2_PER_G * max(0.0, biomass_g)) / PLANT_SPACING_M2
    return max(0.0, lai)

def daily_water(day, light_dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, biomass_g_per_plant=5.0):
    # Penman–Monteith–lite
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    vpd = max(0.05, vpd_kpa(temp_f, rh_frac))  # kPa
    s_slope = 4098 * (0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))) / ((t_c + 237.3) ** 2)  # kPa/°C
    # Net radiation from DLI
    rad_mj = light_dli * RAD_MJ_PER_MOL_PAR * NET_RAD_MULT
    # Canopy coefficient via LAI and stage (stage-weighted to suppress seedling transpiration)
    s_stage = logistic_stage(day)
    lai = _lai_from_biomass(biomass_g_per_plant)
    kc_base = KC_MIN + (KC_MAX - KC_MIN) * (lai / max(1e-9, LAI_MAX))
    kc = np.clip(kc_base * s_stage, 0.0, KC_MAX)

    # FAO-56 ET0-like (mm/day) without soil heat flux and with low wind speed simplification
    num = 0.408 * s_slope * rad_mj + PSYCHROMETRIC_GAMMA * (900.0 / (t_c + 273.0)) * WIND_SPEED * vpd
    den = s_slope + PSYCHROMETRIC_GAMMA * (1.0 + 0.34 * WIND_SPEED)
    et0_mm = max(0.0, num / max(1e-6, den))
    transp_mm = kc * et0_mm

    # Reservoir evaporation (mm/day) from VPD
    evap_mm = EVAP_COEFF_MM_PER_KPA * vpd

    # Convert to liters
    # 1 mm over 1 m^2 equals 1 liter
    transp_l_per_plant = transp_mm * PLANT_SPACING_M2
    evap_l_total = evap_mm * RESERVOIR_AREA_M2
    return 2.0 * transp_l_per_plant + evap_l_total

# Humidity scale mapping from climate via VPD (kPa)
def humidity_scale_from_climate(temp_f, rh_frac, alpha=0.6):
    # Backward-compat shim; daily_water now uses VPD directly. Keep this for uncertainty mapping if needed.
    return float(np.clip((vpd_kpa(temp_f, rh_frac) / max(1e-6, vpd_kpa(AIR_TEMPERATURE, HUMIDITY))) ** alpha, 0.2, 1.8))

# Concentration-dependent uptake function (mg/day/2 plants)
def uptake(c, d, v_max, k_m, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    q10_factor = TEMP_Q10 ** ((temp_f - 77.0) / 18.0)
    stage_factor = logistic_stage(d)
    light_factor = light_saturation(dli)
    vpd_factor = max(0.3, vpd_kpa(temp_f, rh_frac)) ** 0.2  # mild modulation on uptake
    v_max_adj = v_max * stage_factor * light_factor * q10_factor * vpd_factor * ec_factor
    return v_max_adj * c / (k_m + c)

# Specific uptake functions for each nutrient
def uptake_n(c, d, v_max=N_V_MAX, k_m=N_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

def uptake_p(c, d, v_max=P_V_MAX, k_m=P_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

def uptake_k(c, d, v_max=K_V_MAX, k_m=K_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

def uptake_ca(c, d, v_max=CA_V_MAX, k_m=CA_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

def uptake_mg(c, d, v_max=MG_V_MAX, k_m=MG_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

def uptake_fe(c, d, v_max=FE_V_MAX, k_m=FE_K_M, total_days=SIM_DAYS, dli=LIGHT_DLI, temp_f=AIR_TEMPERATURE, rh_frac=HUMIDITY, ec_factor=1.0):
    return uptake(c, d, v_max, k_m, total_days, dli, temp_f, rh_frac, ec_factor)

# Target concentrations (ppm elemental)
def target_n_ppm(d):
    s = logistic_stage(d)
    return 50 + (260 - 50) * s

def target_p_ppm(d):
    s = logistic_stage(d)
    return 20 + (70 - 20) * s

def target_k_ppm(d):
    s = logistic_stage(d)
    return 50 + (235 - 50) * s

def target_ca_ppm(d):
    s = logistic_stage(d)
    return 50 + (170 - 50) * s

def target_mg_ppm(d):
    s = logistic_stage(d)
    return 20 + (56 - 20) * s

def target_fe_ppm(d):
    s = logistic_stage(d)
    return 1 + (3 - 1) * s

# Physiology-aware target construction: scale base targets by ET and biomass
def build_physio_targets(days, daily_w_mean, cum_mass_mean):
    days_arr = np.arange(days + 1)
    base_n = np.array([target_n_ppm(d) for d in days_arr])
    base_p = np.array([target_p_ppm(d) for d in days_arr])
    base_k = np.array([target_k_ppm(d) for d in days_arr])
    base_ca = np.array([target_ca_ppm(d) for d in days_arr])
    base_mg = np.array([target_mg_ppm(d) for d in days_arr])
    base_fe = np.array([target_fe_ppm(d) for d in days_arr])

    # Build normalized ET and biomass signals
    et = np.concatenate([[0.0], np.array(daily_w_mean)])  # align to days+1
    et_norm = et / max(1e-6, np.percentile(et, 90))
    bio = cum_mass_mean
    bio_norm = bio / max(1e-6, bio[-1])

    # Multipliers: modest adjustments only
    n_mult = np.clip(0.95 + 0.20 * bio_norm, 0.9, 1.15)
    p_mult = np.clip(1.00 + 0.10 * (1.0 - bio_norm), 0.95, 1.10)  # a bit higher early
    k_mult = np.clip(1.00 + 0.25 * bio_norm, 0.95, 1.25)
    # Ca: lower target when ET is high to avoid accumulation; slightly higher when ET is low (mass-flow limited)
    ca_mult = np.clip(1.08 - 0.20 * et_norm, 0.85, 1.10)
    mg_mult = np.clip(0.98 + 0.10 * bio_norm, 0.95, 1.12)
    fe_mult = np.clip(1.00 + 0.10 * (1.0 - bio_norm), 0.90, 1.10)

    tgt_n = base_n * n_mult
    tgt_p = base_p * p_mult
    tgt_k = base_k * k_mult
    tgt_ca = base_ca * ca_mult
    tgt_mg = base_mg * mg_mult
    tgt_fe = base_fe * fe_mult

    # Respect toxicity maxima softly by capping targets near max threshold
    tgt_n = np.minimum(tgt_n, TOX_MAX['N'] * 0.98)
    tgt_p = np.minimum(tgt_p, TOX_MAX['P'] * 0.98)
    tgt_k = np.minimum(tgt_k, TOX_MAX['K'] * 0.98)
    tgt_ca = np.minimum(tgt_ca, TOX_MAX['Ca'] * 0.98)
    tgt_mg = np.minimum(tgt_mg, TOX_MAX['Mg'] * 0.98)
    tgt_fe = np.minimum(tgt_fe, TOX_MAX['Fe'] * 0.98)

    return tgt_n, tgt_p, tgt_k, tgt_ca, tgt_mg, tgt_fe

# Simulation function: evolve concentrations with concentration-dependent uptake and param variance
def simulate(ml_init_list, ml_refill_list, days=SIM_DAYS, sample_params=None, light_dli=LIGHT_DLI, air_temp_f=AIR_TEMPERATURE, humidity=HUMIDITY,
             alk_init_pretreat=TAP_ALK_MG_L, alk_refill_pretreat=TAP_ALK_MG_L):
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

    # Initial concentrations (ppm) - fertilizer + tap water background minerals
    init_n, init_p, init_k, init_ca, init_mg, init_fe = chem_nutrient_ppm_from_ml(ml_init_list)
    # Include tap baseline minerals
    init_k += TAP_K_MG_L
    init_ca += TAP_CA_MG_L
    init_mg += TAP_MG_MG_L
    # Apply alkalinity pre-treatment alkalinity change and any acid-contributed nutrients embedded in CHEMICALS
    n_ppm[0] = init_n
    p_ppm[0] = init_p
    k_ppm[0] = init_k
    ca_ppm[0] = init_ca
    mg_ppm[0] = init_mg
    fe_ppm[0] = init_fe

    # Initial pH and alkalinity from pre-treated tap water
    ph = np.full(days + 1, TAP_PH)
    alkalinity = np.full(days + 1, alk_init_pretreat)
    for d in range(days):
        biomass_g_per_plant = (n_uptake_total[d] * n_to_biomass) / 2.0
        W = daily_water(d, light_dli, air_temp_f, humidity, biomass_g_per_plant)

        # Potential uptake (Michaelis-Menten)
        U_n_mm = uptake_n(n_ppm[d], d, n_v_max, n_k_m, days, light_dli, air_temp_f, humidity)
        U_p_mm = uptake_p(p_ppm[d], d, p_v_max, p_k_m, days, light_dli, air_temp_f, humidity)
        U_k_mm = uptake_k(k_ppm[d], d, k_v_max, k_k_m, days, light_dli, air_temp_f, humidity)
        U_ca_mm = uptake_ca(ca_ppm[d], d, ca_v_max, ca_k_m, days, light_dli, air_temp_f, humidity)
        U_mg_mm = uptake_mg(mg_ppm[d], d, mg_v_max, mg_k_m, days, light_dli, air_temp_f, humidity)
        # pH-modulated Fe K_m (reduced availability at higher pH)
        fe_k_m_eff = fe_k_m * (1.0 + PH_FE_SENS * max(0.0, ph[d] - 6.5))
        U_fe_mm = uptake_fe(fe_ppm[d], d, fe_v_max, fe_k_m_eff, days, light_dli, air_temp_f, humidity)

        # Actual N uptake limited by available mass only (demand handled via ratios for others)
        U_n_actual = min(U_n_mm, n_ppm[d] * RESERVOIR_VOLUME)
        n_uptake_total[d + 1] = n_uptake_total[d] + U_n_actual
        mass_n_after = n_ppm[d] * RESERVOIR_VOLUME - U_n_actual
        # N additions from refill chemicals
        refill_n, refill_p, refill_k, refill_ca, refill_mg, refill_fe = chem_nutrient_ppm_from_ml(ml_refill_list)
        n_mass_new = mass_n_after + (refill_n * W)
        n_ppm[d + 1] = max(0, n_mass_new / RESERVOIR_VOLUME)

        # Three-phase stoichiometric ratios relative to N uptake
        phs = phase_for_day(d)
        ratio_p_to_n = PHASE_RATIOS[phs]['P']
        ratio_k_to_n = PHASE_RATIOS[phs]['K']
        ratio_ca_to_n = PHASE_RATIOS[phs]['Ca']
        ratio_mg_to_n = PHASE_RATIOS[phs]['Mg']
        ratio_fe_to_n = PHASE_RATIOS[phs]['Fe']

        demand_p = ratio_p_to_n * U_n_actual
        demand_k = ratio_k_to_n * U_n_actual
        demand_ca = ratio_ca_to_n * U_n_actual
        demand_mg = ratio_mg_to_n * U_n_actual
        demand_fe = ratio_fe_to_n * U_n_actual

        # Actual uptakes capped by MM, demand, and available mass
        U_p_actual = min(U_p_mm, demand_p, p_ppm[d] * RESERVOIR_VOLUME)
        mass_p_after = p_ppm[d] * RESERVOIR_VOLUME - U_p_actual
        p_mass_new = mass_p_after + (refill_p * W)
        p_ppm[d + 1] = max(0, p_mass_new / RESERVOIR_VOLUME)

        U_k_actual = min(U_k_mm, demand_k, k_ppm[d] * RESERVOIR_VOLUME)
        mass_k_after = k_ppm[d] * RESERVOIR_VOLUME - U_k_actual
        k_mass_new = mass_k_after + (refill_k * W)
        k_ppm[d + 1] = max(0, k_mass_new / RESERVOIR_VOLUME)

        # Ca mass-flow cap based on transpiration delivery (W L/day * ppm mg/L)
        ca_mass_flow_cap = W * ca_ppm[d]
        U_ca_actual = min(U_ca_mm, demand_ca, ca_ppm[d] * RESERVOIR_VOLUME, ca_mass_flow_cap)
        mass_ca_after = ca_ppm[d] * RESERVOIR_VOLUME - U_ca_actual
        ca_mass_new = mass_ca_after + (refill_ca * W)
        ca_ppm[d + 1] = max(0, ca_mass_new / RESERVOIR_VOLUME)

        U_mg_actual = min(U_mg_mm, demand_mg, mg_ppm[d] * RESERVOIR_VOLUME)
        mass_mg_after = mg_ppm[d] * RESERVOIR_VOLUME - U_mg_actual
        mg_mass_new = mass_mg_after + (refill_mg * W)
        mg_ppm[d + 1] = max(0, mg_mass_new / RESERVOIR_VOLUME)

        U_fe_actual = min(U_fe_mm, demand_fe, fe_ppm[d] * RESERVOIR_VOLUME)
        mass_fe_after = fe_ppm[d] * RESERVOIR_VOLUME - U_fe_actual
        # Add Fe from refill with decay in prepared refill over time
        fe_add_refill = (refill_fe * W) * ((1.0 - FE_REFILL_DECAY_RATE) ** d)
        fe_mass_new = mass_fe_after + fe_add_refill
        # Apply Fe decay after uptake and refill addition
        fe_mass_new *= (1.0 - FE_DECAY_RATE)
        fe_ppm[d + 1] = max(0, fe_mass_new / RESERVOIR_VOLUME)

        # Simple pH drift: net ion uptake affects alkalinity bidirectionally
        cation_uptake = (U_k_actual + U_ca_actual + U_mg_actual)
        anion_uptake = (U_n_actual + U_p_actual)
        net_acid_eq_mg = anion_uptake - cation_uptake  # positive = acidification, negative = alkalinization
        alk_mass_prev = alkalinity[d] * RESERVOIR_VOLUME
        # Alkalinity changes in response to net ion balance, plus dilution/replacement by refill water
        # Volume W is replaced by refill water with alkalinity alk_refill_pretreat; remaining volume retains previous alkalinity state
        refill_alk_mass = alk_refill_pretreat * W
        remaining_mass = max(0.0, RESERVOIR_VOLUME - W) * alkalinity[d]
        mixed_alk_mass = refill_alk_mass + remaining_mass
        # Apply uptake-driven alkalinity change after mixing
        alk_mass_new = mixed_alk_mass - K_ALK_CONV * net_acid_eq_mg
        alkalinity[d + 1] = max(0.0, alk_mass_new) / max(1e-9, RESERVOIR_VOLUME)
        buffer = 1.0 / (1.0 + ALK_BUFFER_BETA * alkalinity[d])
        ph_delta = buffer * K_PH_DRIFT * ((cation_uptake - anion_uptake) / max(1e-6, RESERVOIR_VOLUME))
        ph[d + 1] = ph[d] + ph_delta - K_PH_RECOVER * (ph[d] - TAP_PH)

    cumulative_mass = n_uptake_total * n_to_biomass / 2  # g dry weight per plant (2 plants)

    return n_ppm, p_ppm, k_ppm, ca_ppm, mg_ppm, fe_ppm, n_uptake_total, cumulative_mass, ph, alkalinity

# Centralized computation of means and variances (±σ propagation of constant climate and uptake parameters)
def compute_means_and_variances(ml_init_list, ml_refill_list, days, sample_params, light_dli, air_temp_f, humidity,
                                alk_init_pt=TAP_ALK_MG_L, alk_refill_pt=TAP_ALK_MG_L):
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

    n_mean, p_mean, k_mean, ca_mean, mg_mean, fe_mean, n_uptake_total, cum_mass_mean, ph_mean, alk_mean = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params, light_dli=light_dli, air_temp_f=air_temp_f, humidity=humidity,
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )

    var_n = np.zeros(days + 1)
    var_p = np.zeros(days + 1)
    var_k = np.zeros(days + 1)
    var_ca = np.zeros(days + 1)
    var_mg = np.zeros(days + 1)
    var_fe = np.zeros(days + 1)

    def add_var_for_params(params_plus, params_minus, ldli, tempf, rh):
        nonlocal var_n, var_p, var_k, var_ca, var_mg, var_fe
        n_p, p_p, k_p, ca_p, mg_p, fe_p, _, _, _, _ = simulate(ml_init_list, ml_refill_list, days=days, sample_params=params_plus, light_dli=ldli, air_temp_f=tempf, humidity=rh,
                                                                alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt)
        n_m, p_m, k_m_, ca_m, mg_m, fe_m, _, _, _, _ = simulate(ml_init_list, ml_refill_list, days=days, sample_params=params_minus, light_dli=ldli, air_temp_f=tempf, humidity=rh,
                                                                 alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt)
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

    # Fe decay rate uncertainty
    p_plus = dict(mean_params); p_minus = dict(mean_params)
    globals()['FE_DECAY_RATE'] = FE_DECAY_RATE + FE_DECAY_RATE_STD
    add_var_for_params(p_plus, p_minus, light_dli, air_temp_f, humidity)
    globals()['FE_DECAY_RATE'] = FE_DECAY_RATE - FE_DECAY_RATE_STD
    add_var_for_params(p_plus, p_minus, light_dli, air_temp_f, humidity)
    globals()['FE_DECAY_RATE'] = FE_DECAY_RATE  # restore

    # Climate parameters (constant across days)
    add_var_for_params(mean_params, mean_params, light_dli + LIGHT_DLI_STD, air_temp_f, humidity)
    add_var_for_params(mean_params, mean_params, max(0.0, light_dli - LIGHT_DLI_STD), air_temp_f, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f + AIR_TEMPERATURE_STD, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f - AIR_TEMPERATURE_STD, humidity)
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f, min(0.99, humidity + HUMIDITY_STD))
    add_var_for_params(mean_params, mean_params, light_dli, air_temp_f, max(0.0, humidity - HUMIDITY_STD))

    # Climate cross-covariance using finite-difference sensitivities
    sig_dli = LIGHT_DLI_STD
    sig_t = AIR_TEMPERATURE_STD
    sig_rh = HUMIDITY_STD
    # DLI sensitivities
    n_dli_p, p_dli_p, k_dli_p, ca_dli_p, mg_dli_p, fe_dli_p, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=light_dli + sig_dli, air_temp_f=air_temp_f, humidity=humidity,
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    n_dli_m, p_dli_m, k_dli_m, ca_dli_m, mg_dli_m, fe_dli_m, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=max(0.0, light_dli - sig_dli), air_temp_f=air_temp_f, humidity=humidity,
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    s_n_dli = (n_dli_p - n_dli_m) / (2.0 * max(1e-9, sig_dli))
    s_p_dli = (p_dli_p - p_dli_m) / (2.0 * max(1e-9, sig_dli))
    s_k_dli = (k_dli_p - k_dli_m) / (2.0 * max(1e-9, sig_dli))
    s_ca_dli = (ca_dli_p - ca_dli_m) / (2.0 * max(1e-9, sig_dli))
    s_mg_dli = (mg_dli_p - mg_dli_m) / (2.0 * max(1e-9, sig_dli))
    s_fe_dli = (fe_dli_p - fe_dli_m) / (2.0 * max(1e-9, sig_dli))

    # Temperature sensitivities
    n_t_p, p_t_p, k_t_p, ca_t_p, mg_t_p, fe_t_p, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=light_dli, air_temp_f=air_temp_f + sig_t, humidity=humidity,
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    n_t_m, p_t_m, k_t_m, ca_t_m, mg_t_m, fe_t_m, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=light_dli, air_temp_f=air_temp_f - sig_t, humidity=humidity,
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    s_n_t = (n_t_p - n_t_m) / (2.0 * max(1e-9, sig_t))
    s_p_t = (p_t_p - p_t_m) / (2.0 * max(1e-9, sig_t))
    s_k_t = (k_t_p - k_t_m) / (2.0 * max(1e-9, sig_t))
    s_ca_t = (ca_t_p - ca_t_m) / (2.0 * max(1e-9, sig_t))
    s_mg_t = (mg_t_p - mg_t_m) / (2.0 * max(1e-9, sig_t))
    s_fe_t = (fe_t_p - fe_t_m) / (2.0 * max(1e-9, sig_t))

    # Humidity sensitivities
    n_rh_p, p_rh_p, k_rh_p, ca_rh_p, mg_rh_p, fe_rh_p, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=light_dli, air_temp_f=air_temp_f, humidity=min(0.99, humidity + sig_rh),
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    n_rh_m, p_rh_m, k_rh_m, ca_rh_m, mg_rh_m, fe_rh_m, _, _, _, _ = simulate(
        ml_init_list, ml_refill_list, days=days, sample_params=mean_params,
        light_dli=light_dli, air_temp_f=air_temp_f, humidity=max(0.0, humidity - sig_rh),
        alk_init_pretreat=alk_init_pt, alk_refill_pretreat=alk_refill_pt
    )
    s_n_rh = (n_rh_p - n_rh_m) / (2.0 * max(1e-9, sig_rh))
    s_p_rh = (p_rh_p - p_rh_m) / (2.0 * max(1e-9, sig_rh))
    s_k_rh = (k_rh_p - k_rh_m) / (2.0 * max(1e-9, sig_rh))
    s_ca_rh = (ca_rh_p - ca_rh_m) / (2.0 * max(1e-9, sig_rh))
    s_mg_rh = (mg_rh_p - mg_rh_m) / (2.0 * max(1e-9, sig_rh))
    s_fe_rh = (fe_rh_p - fe_rh_m) / (2.0 * max(1e-9, sig_rh))

    def add_cross(var_arr, s_dli, s_t, s_rh):
        return var_arr + (
            2.0 * CORR_T_DLI * sig_t * sig_dli * s_t * s_dli +
            2.0 * CORR_DLI_RH * sig_dli * sig_rh * s_dli * s_rh +
            2.0 * CORR_T_RH * sig_t * sig_rh * s_t * s_rh
        )

    var_n = add_cross(var_n, s_n_dli, s_n_t, s_n_rh)
    var_p = add_cross(var_p, s_p_dli, s_p_t, s_p_rh)
    var_k = add_cross(var_k, s_k_dli, s_k_t, s_k_rh)
    var_ca = add_cross(var_ca, s_ca_dli, s_ca_t, s_ca_rh)
    var_mg = add_cross(var_mg, s_mg_dli, s_mg_t, s_mg_rh)
    var_fe = add_cross(var_fe, s_fe_dli, s_fe_t, s_fe_rh)

    # Water: daily and cumulative
    # Biomass proxy derived from mean N uptake trajectory within this call
    # First simulate once with mean parameters and zero-variance to get N uptake trajectory
    mean_params = {
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    # Use a modest, fixed dosing proxy to get a shape for N uptake; absolute scale is not used directly
    num_chems = len(CHEMICALS)
    proxy_init = [0.5]*num_chems
    proxy_refill = [0.5]*num_chems
    n_tmp, _, _, _, _, _, n_uptake_total_tmp, _, _, _ = simulate(proxy_init, proxy_refill, days=days,
        sample_params=mean_params, light_dli=light_dli, air_temp_f=air_temp_f, humidity=humidity,
        alk_init_pretreat=TAP_ALK_MG_L, alk_refill_pretreat=TAP_ALK_MG_L)
    biomass_proxy = (n_uptake_total_tmp / max(1e-9, n_uptake_total_tmp[-1])) * 40.0  # scale to plausible 0-40 g
    biomass_proxy = np.nan_to_num(biomass_proxy)
    daily_w_mean = np.array([daily_water(d, light_dli, air_temp_f, humidity, biomass_proxy[d]) for d in range(days)])
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
    # Add cross-covariance terms for daily water using sensitivities
    s_w_dli = (np.array(daily_w_dli_plus) - np.array(daily_w_dli_minus)) / (2.0 * max(1e-9, sig_dli))
    s_w_t = (np.array(daily_w_temp_plus) - np.array(daily_w_temp_minus)) / (2.0 * max(1e-9, sig_t))
    s_w_rh = (np.array(daily_w_rh_plus) - np.array(daily_w_rh_minus)) / (2.0 * max(1e-9, sig_rh))
    daily_w_var += 2.0 * CORR_T_DLI * sig_t * sig_dli * s_w_t * s_w_dli \
                   + 2.0 * CORR_DLI_RH * sig_dli * sig_rh * s_w_dli * s_w_rh \
                   + 2.0 * CORR_T_RH * sig_t * sig_rh * s_w_t * s_w_rh
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
    # Add cross-covariance terms for cumulative water using sensitivities
    s_cw_dli = (np.array(C_dli_plus) - np.array(C_dli_minus)) / (2.0 * max(1e-9, sig_dli))
    s_cw_t = (np.array(C_temp_plus) - np.array(C_temp_minus)) / (2.0 * max(1e-9, sig_t))
    s_cw_rh = (np.array(C_rh_plus) - np.array(C_rh_minus)) / (2.0 * max(1e-9, sig_rh))
    cum_water_var += 2.0 * CORR_T_DLI * sig_t * sig_dli * s_cw_t * s_cw_dli \
                     + 2.0 * CORR_DLI_RH * sig_dli * sig_rh * s_cw_dli * s_cw_rh \
                     + 2.0 * CORR_T_RH * sig_t * sig_rh * s_cw_t * s_cw_rh

    return {
        'mean': {'N': n_mean, 'P': p_mean, 'K': k_mean, 'Ca': ca_mean, 'Mg': mg_mean, 'Fe': fe_mean},
        'var': {'N': var_n, 'P': var_p, 'K': var_k, 'Ca': var_ca, 'Mg': var_mg, 'Fe': var_fe},
        'n_uptake_total': n_uptake_total,
        'cum_mass_mean': cum_mass_mean,
        'ph_mean': ph_mean,
        'alk_mean': alk_mean,
        'daily_water_mean': daily_w_mean,
        'daily_water_var': daily_w_var,
        'cum_water_mean': cum_water_mean,
        'cum_water_var': cum_water_var,
    }

# Joint optimization bounds for [ml_a_init, ml_b_init, ml_a_refill, ml_b_refill]
def build_bounds():
    chem_bounds = []
    for chem in CHEMICALS:
        lo, hi = chem.get('ml_bounds', (0.0, 10.0))
        chem_bounds.append((lo, hi))
    # bounds for refill chems mirror initial
    # Alkalinity residual bounds adapt to source water; allow reduction down to 0 and up to at least 10 mg/L
    alk_hi = max(10.0, TAP_ALK_MG_L)
    all_bounds = chem_bounds + chem_bounds + [(0.0, alk_hi), (0.0, alk_hi)]
    return all_bounds

# Objective for refill: probabilistic health with analytical uncertainty
def objective_refill(x, sim_days=SIM_DAYS, light_dli=LIGHT_DLI, air_temp_f=AIR_TEMPERATURE, humidity=HUMIDITY):
    init_ml, refill_ml, alk_init_pt, alk_refill_pt = _split_x(x)
    # Use centralized means/variances (also needed for physio-aware targets)
    mean_params = {
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    results = compute_means_and_variances(init_ml, refill_ml, sim_days, mean_params, light_dli, air_temp_f, humidity,
                                          alk_init_pt=alk_init_pt, alk_refill_pt=alk_refill_pt)
    n_sim = results['mean']['N']; p_sim = results['mean']['P']; k_sim = results['mean']['K']
    ca_sim = results['mean']['Ca']; mg_sim = results['mean']['Mg']; fe_sim = results['mean']['Fe']
    var_n = results['var']['N']; var_p = results['var']['P']; var_k = results['var']['K']
    var_ca = results['var']['Ca']; var_mg = results['var']['Mg']; var_fe = results['var']['Fe']
    # Physio-aware targets driven by ET and biomass
    tgt_n, tgt_p, tgt_k, tgt_ca, tgt_mg, tgt_fe = build_physio_targets(sim_days, results['daily_water_mean'], results['cum_mass_mean'])
    target_n = tgt_n; target_p = tgt_p; target_k = tgt_k; target_ca = tgt_ca; target_mg = tgt_mg; target_fe = tgt_fe

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
    # pH Gaussian penalty
    ph = results['ph_mean']
    ph_var = PH_STD ** 2
    # Center pH penalty on plant-optimal PH_BASE (e.g., 6.0)
    loss_ph = np.mean(((ph - PH_BASE) ** 2) / (ph_var + 1e-9) + np.log(ph_var + 1e-9))

    # EC soft guard: estimate EC from major ions (heuristic)
    # Assume TDS ppm ~ 2.0 * (N+P+K+Ca+Mg) to include counter-ions; EC (mS/cm) ~ TDS/640
    tds_ppm = 2.0 * (n_sim + p_sim + k_sim + ca_sim + mg_sim)
    ec_ms = tds_ppm / 640.0
    # Widen acceptable EC band slightly to avoid over-penalizing dosing under hard tap water
    ec_low, ec_high = 1.0, 2.7
    under = np.maximum(0.0, ec_low - ec_ms)
    over = np.maximum(0.0, ec_ms - ec_high)
    ec_penalty = np.mean(under**2 + over**2)
    LAMBDA_EC = 0.5

    # Asymmetric toxicity penalties (soft upper bounds)
    TOX_MAX = {'N': 300.0, 'P': 100.0, 'K': 300.0, 'Ca': 190.0, 'Mg': 70.0, 'Fe': 4.0}
    tox_n = np.mean(np.maximum(0.0, n_sim - TOX_MAX['N'])**2)
    tox_p = np.mean(np.maximum(0.0, p_sim - TOX_MAX['P'])**2)
    tox_k = np.mean(np.maximum(0.0, k_sim - TOX_MAX['K'])**2)
    tox_ca = np.mean(np.maximum(0.0, ca_sim - TOX_MAX['Ca'])**2)
    tox_mg = np.mean(np.maximum(0.0, mg_sim - TOX_MAX['Mg'])**2)
    tox_fe = np.mean(np.maximum(0.0, fe_sim - TOX_MAX['Fe'])**2)
    LAMBDA_TOX = 0.01

    # Mild regularization preferring some alkalinity reduction (to enable more Part A) but avoiding over-acidification
    alk_series = results['alk_mean']
    LAMBDA_ALK = 1e-4
    alk_reg = LAMBDA_ALK * np.mean((np.maximum(0.0, TAP_ALK_MG_L - alk_series))**2)

    # Asymmetric shortfall penalty to avoid degenerate zero-dosing solutions
    def rel_shortfall(sim, tgt):
        return np.mean((np.maximum(0.0, tgt - sim) / (tgt + 1e-9))**2)
    LAMBDA_SHORT = 10.0
    short_pen = LAMBDA_SHORT * (
        4.0 * rel_shortfall(n_sim, target_n) +
        1.5 * rel_shortfall(p_sim, target_p) +
        3.0 * rel_shortfall(k_sim, target_k) +
        2.5 * rel_shortfall(ca_sim, target_ca)
    )

    return (loss_n + loss_p + loss_k + loss_ca + loss_mg + loss_fe + loss_ph + alk_reg + short_pen
            + LAMBDA_EC * ec_penalty
            + LAMBDA_TOX * (tox_n + tox_p + tox_k + tox_ca + tox_mg + tox_fe))

def integrated_target_masses(days, daily_w_mean, cum_mass_mean):
    tgt_n, tgt_p, tgt_k, tgt_ca, _, _ = build_physio_targets(days, daily_w_mean, cum_mass_mean)
    # integrate by simple sum (per-day ppm proxy); scale to mg mass target by reservoir volume if needed
    return np.sum(tgt_n), np.sum(tgt_p), np.sum(tgt_k), np.sum(tgt_ca)

def inequality_constraints(sim_days, light_dli, air_temp_f, humidity):
    # Build inequality constraint functions c(x) >= 0
    def constr_fun(x):
        init_ml, refill_ml, alk_init_pt, alk_refill_pt = _split_x(x)
        mean_params = {
            'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
            'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
            'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
            'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
            'n_to_biomass': N_TO_BIOMASS,
        }
        res = compute_means_and_variances(init_ml, refill_ml, sim_days, mean_params, light_dli, air_temp_f, humidity,
                                          alk_init_pt=alk_init_pt, alk_refill_pt=alk_refill_pt)
        n_mean = res['mean']['N']; p_mean = res['mean']['P']; k_mean = res['mean']['K']; ca_mean = res['mean']['Ca']
        # approximate uptake mass via tracking to target area; use integrated targets as minimum coverage
        M_n, M_p, M_k, M_ca = integrated_target_masses(sim_days, res['daily_water_mean'], res['cum_mass_mean'])
        cov_n = np.sum(n_mean)
        cov_p = np.sum(p_mean)
        cov_k = np.sum(k_mean)
        cov_ca = np.sum(ca_mean)
        alpha_n, alpha_p, alpha_k, alpha_ca = 0.65, 0.55, 0.60, 0.55
        # EC / pH constraints (average within band)
        tds_ppm = 2.0 * (n_mean + p_mean + k_mean + ca_mean + res['mean']['Mg'])
        ec_ms = tds_ppm / 640.0
        ec_ok = 2.0 - np.abs(np.mean(ec_ms) - 1.8)  # >=0 when mean within ~[1.1,2.5]
        ph_ok = 0.3 - np.abs(np.mean(res['ph_mean']) - PH_BASE)  # >=0 if mean within ±0.3
        return np.array([
            cov_n - alpha_n * M_n,
            cov_p - alpha_p * M_p,
            cov_k - alpha_k * M_k,
            cov_ca - alpha_ca * M_ca,
            ec_ok,
            ph_ok,
        ])
    return constr_fun

def main():
    global RISK_LAMBDA  # Declare at function start
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

    RISK_LAMBDA = args.risk_lambda

    # Jointly optimize [ml_a_init, ml_b_init, ml_a_refill, ml_b_refill]
    num_chems = len(CHEMICALS)
    # initial guess: 1 ml/L for each chem in init and refill, alkalinity ~60% of tap
    x0 = [1.0] * (num_chems * 2) + [max(40.0, TAP_ALK_MG_L * 0.6), max(40.0, TAP_ALK_MG_L * 0.6)]
    cons_fun = inequality_constraints(SIM_DAYS, light_dli, air_temp, rh)
    constraints = ({'type': 'ineq', 'fun': cons_fun},)
    result = minimize(lambda x: objective_refill(x, SIM_DAYS, light_dli, air_temp, rh), x0, bounds=build_bounds(), method='SLSQP', constraints=constraints, options={'maxiter': 200, 'ftol': 1e-4})
    vec = result.x
    init_ml = list(vec[:num_chems])
    refill_ml = list(vec[num_chems:2*num_chems])
    alk_init_pt, alk_refill_pt = vec[-2], vec[-1]
    optimal_n_refill, optimal_p_refill, optimal_k_refill, optimal_ca_refill, optimal_mg_refill, optimal_fe_refill = chem_nutrient_ppm_from_ml(refill_ml)

    # Scale to 4L for initial
    init_ml_4l = [m * RESERVOIR_VOLUME for m in init_ml]

    print("\nOptimization Results (generalized chemical system):")
    for name, ml in zip([c['name'] for c in CHEMICALS], init_ml_4l):
        print(f"  Initial in 4L: {ml:.2f} ml {name}")
    init_n, init_p, init_k, init_ca, init_mg, init_fe = chem_nutrient_ppm_from_ml(init_ml)
    print(f"  Achieves approx. N {init_n:.1f} ppm, P {init_p:.1f} ppm")
    print(f"  K {init_k + TAP_K_MG_L:.1f} ppm (includes tap {TAP_K_MG_L:.1f})")
    print(f"  Ca {init_ca + TAP_CA_MG_L:.1f} ppm (includes tap {TAP_CA_MG_L:.1f})")
    print(f"  Mg {init_mg + TAP_MG_MG_L:.1f} ppm (includes tap {TAP_MG_MG_L:.1f})")
    print(f"  Fe {init_fe:.1f} ppm")
    print("Optimal per-liter refill:")
    for name, ml in zip([c['name'] for c in CHEMICALS], refill_ml):
        print(f"  {name}: {ml:.2f} ml/L")
    print(f"  Provides N {optimal_n_refill:.1f} ppm, P {optimal_p_refill:.1f} ppm, K {optimal_k_refill:.1f} ppm")
    print(f"  Ca {optimal_ca_refill:.1f} ppm, Mg {optimal_mg_refill:.1f} ppm, Fe {optimal_fe_refill:.1f} ppm")
    print(f"  - This constant mix maintains concentrations close to targets over {SIM_DAYS} days.")
    # Acid pre-treatment summary
    acid_init_removed = max(0.0, TAP_ALK_MG_L - alk_init_pt)
    acid_refill_removed = max(0.0, TAP_ALK_MG_L - alk_refill_pt)
    # Estimate N from acid based on declared nitric acid chemical in CHEMICALS
    nitric_index = next((i for i,c in enumerate(CHEMICALS) if 'nitric' in c['name'].lower()), None)
    acid_n_ppm_per_ml = CHEMICALS[nitric_index]['per_ml_ppm'].get('N', 0.0) if nitric_index is not None else 0.0
    acid_alk_per_ml = abs(CHEMICALS[nitric_index].get('alk_change_mg_per_ml', 0.0)) if nitric_index is not None else 1.0
    n_per_mg_caco3 = (acid_n_ppm_per_ml / acid_alk_per_ml) if acid_alk_per_ml > 0 else 0.0
    n_from_acid_init = acid_init_removed * n_per_mg_caco3
    n_from_acid_refill = acid_refill_removed * n_per_mg_caco3
    print(f"Acid pre-treatment: initial alkalinity {TAP_ALK_MG_L:.1f} -> {alk_init_pt:.1f} mg/L (adds ~{n_from_acid_init:.1f} ppm N equivalent)")
    print(f"Refill water alkalinity {TAP_ALK_MG_L:.1f} -> {alk_refill_pt:.1f} mg/L (adds ~{n_from_acid_refill:.1f} ppm N per liter of refill)")
    print(f"Note: Estimates for Thai peppers at 85F, 70% RH; uptake depends on concentration. Monitor EC (1.5-2.5 mS/cm), pH ({TAP_PH-0.4:.1f}-{TAP_PH+0.4:.1f}).")
    print(f"Tap water background: Ca {TAP_CA_MG_L} mg/L, Mg {TAP_MG_MG_L} mg/L, K {TAP_K_MG_L} mg/L, pH {TAP_PH}, Alk {TAP_ALK_MG_L} mg/L CaCO3.")

    # Centralized means and variances for plotting
    mean_params = {
        'n_v_max': N_V_MAX, 'p_v_max': P_V_MAX, 'k_v_max': K_V_MAX,
        'ca_v_max': CA_V_MAX, 'mg_v_max': MG_V_MAX, 'fe_v_max': FE_V_MAX,
        'n_k_m': N_K_M, 'p_k_m': P_K_M, 'k_k_m': K_K_M,
        'ca_k_m': CA_K_M, 'mg_k_m': MG_K_M, 'fe_k_m': FE_K_M,
        'n_to_biomass': N_TO_BIOMASS,
    }
    results = compute_means_and_variances(init_ml, refill_ml, SIM_DAYS, mean_params, light_dli, air_temp, rh,
                                          alk_init_pt=alk_init_pt, alk_refill_pt=alk_refill_pt)
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
    cum_water_mean = np.cumsum(daily_w)
    cum_water_std = np.sqrt(results['cum_water_var'])
    n_uptake_total = results['n_uptake_total']
    cum_mass_mean = results['cum_mass_mean']

# Calculate cumulative water usage and plant mass (mean +/- std)
    days = np.arange(SIM_DAYS + 1)
    print(f"\nCumulative refill water usage over {SIM_DAYS} days: {cum_water_mean[-1]:.1f} ± {cum_water_std[-1]:.1f} L for 2 plants")
    if args.print_interval and args.print_interval > 0:
        step = args.print_interval
        print("\nDay | N PPM | P PPM | K PPM | Ca PPM | Mg PPM | Fe PPM | Daily W (L/2) | Cum W (L/2) | Biomass g/plant")
        for d in range(0, SIM_DAYS + 1, step):
            daily_w_d = daily_w[d-1] if d > 0 and d-1 < len(daily_w) else 0.0
            print(f"{d:3d} | {n_mean[d]:6.1f} | {p_mean[d]:6.1f} | {k_mean[d]:6.1f} | {ca_mean[d]:6.1f} | {mg_mean[d]:6.1f} | {fe_mean[d]:6.2f} | {daily_w_d:7.3f} | {cum_water_mean[d-1] if d>0 else 0.0:7.3f} | {cum_mass_mean[d]:6.2f}")
    # Biomass uncertainty via delta method: var(mass) ≈ (N_TO_BIOMASS_STD * uptake)^2 + (N_TO_BIOMASS * uptake_std)^2
    # Approximate uptake_std from nutrient concentration variance proxy (N) using centralized results
    uptake_std = np.sqrt(np.maximum(0.0, results['var']['N'][-1]))
    mass_std = np.sqrt((N_TO_BIOMASS_STD * n_uptake_total[-1])**2 + (N_TO_BIOMASS * uptake_std)**2) / 2.0
    print(f"Cumulative plant mass (dry weight) over {SIM_DAYS} days: {cum_mass_mean[-1]:.1f} ± {mass_std:.1f} g per plant")

    if not args.no_plot:
# Plot simulation with error bar shading (mean +/- 1 std)
        target_n, target_p, target_k, target_ca, target_mg, target_fe = build_physio_targets(
            SIM_DAYS, results['daily_water_mean'], results['cum_mass_mean']
        )

        # Plot target Gaussian bands using same TGT_REL/TGT_ABS as objective (95% CI)
        def tgt_std(arr_target, key):
            return np.maximum(TGT_REL[key] * arr_target, TGT_ABS[key])
        n_band = CI_Z * tgt_std(target_n, 'N')
        p_band = CI_Z * tgt_std(target_p, 'P')
        k_band = CI_Z * tgt_std(target_k, 'K')
        ca_band = CI_Z * tgt_std(target_ca, 'Ca')
        mg_band = CI_Z * tgt_std(target_mg, 'Mg')
        fe_band = CI_Z * tgt_std(target_fe, 'Fe')

        # 95% CI for simulated concentrations
        n_ci = CI_Z * n_std
        p_ci = CI_Z * p_std
        k_ci = CI_Z * k_std
        ca_ci = CI_Z * ca_std
        mg_ci = CI_Z * mg_std
        fe_ci = CI_Z * fe_std

        fig, axs = plt.subplots(5, 2, figsize=(15, 20))
        fig.suptitle('Hydroponic Nutrient Simulation Results', fontsize=16, fontweight='bold', y=0.98)
        axs[0,0].fill_between(days, target_n - n_band, target_n + n_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[0,0].fill_between(days, target_n, np.minimum(TOX_MAX['N'], target_n), color='none')
        axs[0,0].axhline(TOX_MAX['N'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[0,0].plot(days, target_n, label='Target N PPM', linestyle='--', color='gray')
        axs[0,0].plot(days, n_mean, label='Mean Simulated N PPM')
        axs[0,0].fill_between(days, n_mean - n_ci, n_mean + n_ci, alpha=0.3, label='95% CI (analytic)')
        axs[0,0].set_ylabel('N (ppm)')
        axs[0,0].set_title('Nitrogen')
        axs[0,0].legend()
        axs[0,0].grid(True, alpha=0.3)

        axs[0,1].fill_between(days, target_p - p_band, target_p + p_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[0,1].axhline(TOX_MAX['P'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[0,1].plot(days, target_p, label='Target P PPM', linestyle='--', color='gray')
        axs[0,1].plot(days, p_mean, label='Mean Simulated P PPM')
        axs[0,1].fill_between(days, p_mean - p_ci, p_mean + p_ci, alpha=0.3, label='95% CI (analytic)')
        axs[0,1].set_ylabel('P (ppm)')
        axs[0,1].set_title('Phosphorus')
        axs[0,1].legend()
        axs[0,1].grid(True, alpha=0.3)

        axs[1,0].fill_between(days, target_k - k_band, target_k + k_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[1,0].axhline(TOX_MAX['K'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[1,0].plot(days, target_k, label='Target K PPM', linestyle='--', color='gray')
        axs[1,0].plot(days, k_mean, label='Mean Simulated K PPM')
        axs[1,0].fill_between(days, k_mean - k_ci, k_mean + k_ci, alpha=0.3, label='95% CI (analytic)')
        axs[1,0].set_ylabel('K (ppm)')
        axs[1,0].set_title('Potassium')
        axs[1,0].legend()
        axs[1,0].grid(True, alpha=0.3)

        axs[1,1].fill_between(days, target_ca - ca_band, target_ca + ca_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[1,1].axhline(TOX_MAX['Ca'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[1,1].plot(days, target_ca, label='Target Ca PPM', linestyle='--', color='gray')
        axs[1,1].plot(days, ca_mean, label='Mean Simulated Ca PPM')
        axs[1,1].fill_between(days, ca_mean - ca_ci, ca_mean + ca_ci, alpha=0.3, label='95% CI (analytic)')
        axs[1,1].set_ylabel('Ca (ppm)')
        axs[1,1].set_title('Calcium')
        axs[1,1].legend()
        axs[1,1].grid(True, alpha=0.3)

        axs[2,0].fill_between(days, target_mg - mg_band, target_mg + mg_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[2,0].axhline(TOX_MAX['Mg'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[2,0].plot(days, target_mg, label='Target Mg PPM', linestyle='--', color='gray')
        axs[2,0].plot(days, mg_mean, label='Mean Simulated Mg PPM')
        axs[2,0].fill_between(days, mg_mean - mg_ci, mg_mean + mg_ci, alpha=0.3, label='95% CI (analytic)')
        axs[2,0].set_ylabel('Mg (ppm)')
        axs[2,0].set_title('Magnesium')
        axs[2,0].legend()
        axs[2,0].grid(True, alpha=0.3)

        axs[2,1].fill_between(days, target_fe - fe_band, target_fe + fe_band, color='orange', alpha=0.15, label='Target 95% CI')
        axs[2,1].axhline(TOX_MAX['Fe'], color='red', linestyle=':', alpha=0.6, label='Toxicity max')
        axs[2,1].plot(days, target_fe, label='Target Fe PPM', linestyle='--', color='gray')
        axs[2,1].plot(days, fe_mean, label='Mean Simulated Fe PPM')
        axs[2,1].fill_between(days, fe_mean - fe_ci, fe_mean + fe_ci, alpha=0.3, label='95% CI (analytic)')
        axs[2,1].set_ylabel('Fe (ppm)')
        axs[2,1].set_title('Iron')
        axs[2,1].legend()
        axs[2,1].grid(True, alpha=0.3)

        axs[3,0].plot(days[1:], cum_water_mean, label='Mean Cumulative Water Use (L for 2 plants)', color='purple')
        axs[3,0].fill_between(days[1:], cum_water_mean - CI_Z * cum_water_std, cum_water_mean + CI_Z * cum_water_std, alpha=0.3, color='purple')
        axs[3,0].plot(days, cum_mass_mean, label='Mean Cumulative Plant Mass (g dry weight, per plant)', color='green')
        axs[3,0].set_ylabel('Cumulative Water (L) / Mass (g)')
        axs[3,0].set_title('Water Use & Plant Mass')
        axs[3,0].legend(loc='upper left')
        axs[3,0].grid(True, alpha=0.3)

        # Add daily water use on a separate axis
        ax30_twin = axs[3,0].twinx()
        ax30_twin.plot(days[:SIM_DAYS], daily_w, label='Daily Water Use (L for 2 plants)', color='blue', linestyle='--', alpha=0.8)
        ax30_twin.fill_between(days[:SIM_DAYS], daily_w - CI_Z * daily_w_std, daily_w + CI_Z * daily_w_std, alpha=0.3, color='blue')
        ax30_twin.set_ylabel('Daily Water Use (L)', color='blue')
        ax30_twin.tick_params(axis='y', labelcolor='blue')

        # Combine legends from both axes
        lines1, labels1 = axs[3,0].get_legend_handles_labels()
        lines2, labels2 = ax30_twin.get_legend_handles_labels()
        axs[3,0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Calculate EC time series
        ec_mean = 2.0 * (n_mean + p_mean + k_mean + ca_mean + mg_mean) / 640.0

        # EC plot
        axs[3,1].plot(days, ec_mean, label='Electrical Conductivity', color='brown')
        axs[3,1].axhline(2.5, color='red', linestyle=':', alpha=0.6, label='High EC threshold')
        axs[3,1].axhline(1.2, color='orange', linestyle='--', alpha=0.6, label='Low EC threshold')
        axs[3,1].set_ylabel('EC (mS/cm)')
        axs[3,1].set_xlabel('Days')
        axs[3,1].set_title('Electrical Conductivity')
        axs[3,1].legend()
        axs[3,1].grid(True, alpha=0.3)

        # pH plot
        axs[4,0].plot(days, results['ph_mean'], label='pH', color='magenta')
        axs[4,0].axhline(TAP_PH + 0.4, color='red', linestyle=':', alpha=0.6, label=f'pH max ({TAP_PH + 0.4:.1f})')
        axs[4,0].axhline(TAP_PH - 0.4, color='orange', linestyle='--', alpha=0.6, label=f'pH min ({TAP_PH - 0.4:.1f})')
        axs[4,0].axhline(TAP_PH, color='blue', linestyle='-', alpha=0.3, label=f'Tap water pH ({TAP_PH})')
        axs[4,0].set_ylabel('pH')
        axs[4,0].set_title('pH Level')
        axs[4,0].legend()
        axs[4,0].grid(True, alpha=0.3)

        # Alkalinity plot
        axs[4,1].plot(days, results['alk_mean'], label='Alkalinity', color='cyan')
        axs[4,1].axhline(TAP_ALK_MG_L, color='blue', linestyle='-', alpha=0.3, label=f'Tap water ({TAP_ALK_MG_L} mg/L)')
        axs[4,1].set_ylabel('Alkalinity (mg/L CaCO3)')
        axs[4,1].set_xlabel('Days')
        axs[4,1].set_title('Alkalinity')
        axs[4,1].legend()
        axs[4,1].grid(True, alpha=0.3)


        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
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
    parser.add_argument('--print-interval', type=int, default=0,
                       help='Print debug metrics every N days (0 to disable)')
    return parser.parse_args()

if __name__ == '__main__':
    main()
