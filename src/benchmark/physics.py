import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# ===============================================================
# CONSTANTES GLOBALES DE RÉFÉRENCE 
# ===============================================================
Rg = 1.98717           # Constante des gaz en cal/mol/K
T_REF = 649.15         # Température de référence (K)
P_REF = 120.0          # Pression de référence en H2 (bar)
RES_REF = 2.0e5        # Référence de concentration en résine (ppm)
N_REF = 5000.0         # Concentration de référence en azote (ppm)
S_REF = 2.5            # Concentration de référence en soufre (ppm)
TMP_REF = 650.0        # Température de référence pour TMP (K)
PPNH3_REF = 0.25       # Pression partielle de NH3 de référence (bar)

# ===============================================================
# PARAMÈTRES CINÉTIQUES DU MODÈLE HDN (HydroDénitrogénation)
# ===============================================================
k0_N = 3.431           # Constante pré-exponentielle HDN
Ea_N = 31015           # Énergie d'activation (cal/mol)
m_N = 0.650            # Exposant de la pression H2
n_N = 1.058            # Ordre par rapport à N
u_N = 0.714            # Coefficient du terme d’équilibre
aret = -0.191          # Exposant dans l'équation d'équilibre (pression)
b_ret = 1936.7         # Constante d’équilibre (K)
alpha8 = -0.488        # Influence du ratio N0/S0
alpha9 = -0.715        # Influence de T05
w1, w2, w3 = -0.026, 0.100, -0.150  # Exposants pour l'équilibre HDN
p_NH3 = -0.128         # Influence de la pression partielle de NH3

# ===============================================================
# PARAMÈTRES CINÉTIQUES DU MODÈLE HDS (HydroDésulfuration)
# ===============================================================
k0_S = 0.03060         # Constante pré-exponentielle HDS
Ea_S = 34467           # Énergie d’activation (cal/mol)
m_S = 1.611            # Ordre par rapport à S
m1, m2 = 0.714, -0.490 # Exposants de H2 et H2S
wS1, wS6, wS7 = -0.349, -0.390, -0.817  # Exposants dans DS
pS_NH3 = -0.125        # Influence de la pression partielle de NH3

# ===============================================================
# Fonction d’équilibre thermodynamique pour HDN
# ===============================================================
def k_eq(T, ppH2, Nppm, Res, TMP):
    """
    Calcule le facteur d'équilibre k_eq pour le modèle HDN.

    T : Température (K)
    ppH2 : Pression partielle d'hydrogène (bar)
    Nppm : Concentration en azote (ppm)
    Res : Charge en résine (ppm)
    TMP : Température TMP (K)
    """
    return (
        u_N *
        (ppH2 / P_REF)**aret *
        np.exp(-(b_ret / Rg) * (1/T - 1/T_REF)) *
        (Nppm / N_REF)**w1 *
        (Res / RES_REF)**w2 *
        (TMP / TMP_REF)**w3
    )

# ===============================================================
# ÉQUATIONS DIFFÉRENTIELLES  POUR HDN & HDS
# ===============================================================
def system(t, y, T_K, T05, TMP, ppH2, ppH2S, ppNH3, N0_ppm, S0_ppm, Res):
    """
    Système d'équations différentielles décrivant les cinétiques HDN et HDS.
    
    y[0] : N(t) (ppm)
    y[1] : S(t) (ppm)
    """
    Nppm = max(y[0], 1e-4)  # Évite les valeurs négatives
    Sppm = max(y[1], 1e-4)

    # ===== HDN =====
    termN = (
        -k0_N *
        np.exp(-Ea_N / Rg * (1/T_K - 1/T_REF)) *
        (ppH2 / P_REF)**m_N *
        (ppNH3 / PPNH3_REF)**p_NH3
    )

    DN = (((N0_ppm * 1e4) / S0_ppm) / (N_REF / S_REF))**alpha8 * (T05 / TMP_REF)**alpha9

    dN = termN * (1 - k_eq(T_K, ppH2, Nppm, Res, TMP)) * Nppm**n_N * DN

    # ===== HDS =====
    termS = (
        -k0_S *
        (ppH2 / P_REF)**m1 *
        ppH2S**m2 *
        (ppNH3 / PPNH3_REF)**pS_NH3 *
        np.exp(-Ea_S / Rg * (1/T_K - 1/T_REF))
    )

    DS = (
        (Nppm / N_REF)**wS1 *
        (TMP / TMP_REF)**wS6 *
        (T05 / TMP_REF)**wS7
    )

    dSppm = termS * Sppm**m_S * DS

    return [dN, dSppm]

# ===============================================================
# Fonction de simulation pour une ligne du DataFrame
# ===============================================================
def simulate_row(row):
    """
    Simule les valeurs de N et S en sortie à partir d’une ligne de données.
    
    Paramètres :
    row : pd.Series – Une ligne de données expérimentales avec au minimum les colonnes :
          T_k, T05, TMP, ppH2, ppH2S, ppNH3, N0, S0, Resines_chg, t

    Retourne :
    [N_sim, S_sim] à t = row.t
    """
    args = (
        row.T_k, row.T05, row.TMP,
        row.ppH2, row.ppH2S, row.ppNH3,
        row.N0, row.S0, row.Resines_chg
    )
    
    # solve_ivp intègre entre t=0 et t=row.t, avec les conditions initiales N0, S0
    sol = solve_ivp(
        system,
        (0, row.t),
        y0=[row.N0, row.S0],
        args=args,
        method="LSODA",    # Méthode equivalente à LSODE dans Fortran
        t_eval=[row.t]     # Évalue uniquement à t final
    )
    
    return sol.y[:, 0]  # Retourne les concentrations finales [N, S]
