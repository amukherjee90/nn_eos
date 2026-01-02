import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plot_helper as ph


from pathlib import Path
project_root= Path.cwd()
#print(project_root)
#print(type(str(project_root)))
#data_dir = project_root / "data" / "raw"
#data_dir.mkdir(parents=True, exist_ok=True)
filepath = str(project_root)+"/data/raw/pr_water_rho.csv"
#print(filepath)
df = pd.read_csv(filepath)
print(df.head())


random_P_values = df['P_Pa'].sample(n=4)
print(random_P_values)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex = True, sharey=True)

for ax, P0 in zip(axes.flat, random_P_values):
    ph.plot_P_fixed(ax, P0, df)

plt.tight_layout()

out_dir = project_root / "eos_solver_check" 
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "T_vs_rho.png", dpi=300)
plt.show()
