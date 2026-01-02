
import pandas as pd
import numpy as np


def scaling_save():
    from pathlib import Path
    project_root= Path.cwd()
    filepath_load = str(project_root)+"/data/raw/pr_water_rho.csv"
    #print(filepath)
    rho_orig = pd.read_csv(filepath_load)
    #print(rho_orig.head())
    
    #print(rho_orig.columns)
    
    rho_orig["T_K"] = ((rho_orig["T_K"] - rho_orig["T_K"].min()) /  
                       (rho_orig["T_K"].max()-rho_orig["T_K"].min()))
                      
    rho_orig["P_Pa"] = ((rho_orig["P_Pa"] - rho_orig["P_Pa"].min()) /  
                       (rho_orig["P_Pa"].max()-rho_orig["P_Pa"].min()))
    
    rho_orig["rho_liquid"] = rho_orig["rho_liquid"]/10.0
    rho_orig["rho_vapor"]  = rho_orig["rho_vapor"]/10.0
    
    #print(rho_orig.head())
    
    
    data_dir = project_root / "data" / "scaled"
    #print(type(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath_save = data_dir/ "pr_water_rho_scaled.csv"
    
    rho_orig.to_csv(filepath_save, index = False)
    
def scaling_load(rho_orig):
    rho_orig["T_K"] = ((rho_orig["T_K"] - rho_orig["T_K"].min()) /  
                       (rho_orig["T_K"].max()-rho_orig["T_K"].min()))
                      
    rho_orig["P_Pa"] = ((rho_orig["P_Pa"] - rho_orig["P_Pa"].min()) /  
                       (rho_orig["P_Pa"].max()-rho_orig["P_Pa"].min()))
    
    rho_orig["rho_liquid"] = rho_orig["rho_liquid"]/10.0
    rho_orig["rho_vapor"]  = rho_orig["rho_vapor"]/10.0
    return rho_orig