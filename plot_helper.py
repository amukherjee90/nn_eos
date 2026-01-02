import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_P_fixed(ax,P0,df):
    
    tol = 1e-6 * P0   # relative tolerance

    dfP = df[np.abs(df["P_Pa"] - P0) < tol]
    df_sorted = dfP.sort_values("T_K")
    df_sorted.drop(columns = ["P_Pa"],inplace = True)
    df_liq = df_sorted[df_sorted['rho_liquid']>100.0]
    df_vap = df_sorted[df_sorted['rho_vapor']<100.0]
    
    ax.plot(df_liq["rho_liquid"], df_liq["T_K"])
    ax.plot(df_vap["rho_vapor"], df_vap["T_K"])
    ax.set_xlabel("Density")
    ax.set_ylabel("Temperature")
    #plt.legend()
    ax.set_title(f"P0 = {P0/1e5:.2f} Bar")
    ax.grid()
    

def plot_T_fixed(ax,T0):
    
    tol = 1.0   # relative tolerance

    dfP = df[np.abs(df["T_K"] - T0) < tol]
    df_sorted = dfP.sort_values("P_Pa")
    df_sorted.drop(columns = ["T_K"],inplace = True)
    df_liq = df_sorted[df_sorted['rho_liquid']>100.0]
    df_vap = df_sorted[df_sorted['rho_vapor']<100.0]
    
    ax.plot(df_liq["rho_liquid"], df_liq["P_Pa"])
    ax.plot(df_vap["rho_vapor"], df_vap["P_Pa"])
    ax.set_title(f"T0 = {T0:.2f} K")
    ax.set_xlabel("Density")
    ax.set_ylabel("Pressure")
    #plt.legend()
    ax.grid() 