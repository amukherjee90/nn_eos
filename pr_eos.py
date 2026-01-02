import numpy as np
import pandas as pd
import time

R = 8.314462618  # J/(mol*K)
M_water = 0.01801528  # kg/mol

# --- Water critical properties and acentric factor ---
Tc_water = 647.1        # K
Pc_water = 22.064e6     # Pa
omega_water = 0.344

# --- Peng-Robinson parameters for water ---
a_water = 0.45724 * R**2 * Tc_water**2 / Pc_water
b_water = 0.07780 * R * Tc_water / Pc_water
kappa_water = 0.37464 + 1.54226 * omega_water - 0.26992 * omega_water**2

def alpha_PR(T, Tc=Tc_water, kappa=kappa_water):
    Tr = T / Tc
    return (1 + kappa * (1 - np.sqrt(Tr)))**2


def Z_roots_PR_water(T, P):
    alpha = alpha_PR(T)
    A = a_water * alpha * P / (R**2 * T**2)
    B = b_water * P / (R * T)

    c2 = -(1 - B)
    c1 = A - 3*B**2 - 2*B
    c0 = -(A*B - B**2 - B**3)

    coeffs = np.array([1.0, c2, c1, c0], dtype=float)
    return np.roots(coeffs)

"""
Cubic EOS ca return 
1. 3 real roots - two-phase
2. 1 real, 2 complex roots - single phase

roots can be something like this -
array([0.031+0.j, 0.872+0.j, 0.093+0.j])
"""

def real_Z_roots(T, P, tol=1e-8):
    roots = Z_roots_PR_water(T, P)
   # print(roots)
#this returns Boolean values - True for real and False for iamg
    mask_real = np.abs(roots.imag) < tol
  #  print(mask_real)
  #  print(roots.real[mask_real])
#only keeps the real roots, where it encounters True
    return np.sort(roots.real[mask_real])

def density_from_Z(T, P, Z, M=M_water):
    """
    Return mass density [kg/m^3] from PR Z-factor.
    T [K], P [Pa], Z dimensionless.
    """
    return P * M / (Z * R * T)


def densities_PR_water(T, P, tol=1e-8):
    
    """
    Return (rho_liquid, rho_vapor) for given T [K], P [Pa].
    If only one real root, returns (rho_single, None).
    If no real roots, returns (None, None).
    """
    z_real = real_Z_roots(T, P, tol=tol)
  #  print(type(z_real),z_real.shape)
    if len(z_real) == 0:
        return None, None
    elif len(z_real) == 1:
        rho = density_from_Z(T, P, z_real[0])
        rho_liq = rho
        rho_vap = rho
        return rho_liq, rho_vap
    else:
        Z_liq = z_real[0]
        Z_vap = z_real[-1]
        rho_liq = density_from_Z(T, P, Z_liq)
        rho_vap = density_from_Z(T, P, Z_vap)
        return rho_liq, rho_vap


T_vals = np.linspace(400, 630, 100)        
P_vals = np.linspace(1e5, 2e7, 100)       

rows = []
    

start = time.perf_counter()

for P in P_vals:
    for T in T_vals:
        
        rho_liq, rho_vap = densities_PR_water(T, P)
        
        rows.append({
            "T_K": T,
            "P_Pa": P,
            "rho_liquid": rho_liq,
            "rho_vapor": rho_vap
        })
        
end = time.perf_counter()

duration = end -start

from pathlib import Path
project_root= Path.cwd()
data_dir = project_root / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)
filepath = data_dir/ "pr_water_rho.csv"

df = pd.DataFrame(rows)
df.to_csv(filepath, index = False)
print(duration)

    
