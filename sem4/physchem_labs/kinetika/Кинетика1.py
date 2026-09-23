import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

file_path = '/Users/vasilij/Documents/Python/.conda/Физхимия/Кинетика1.csv'
df = pd.read_csv(file_path, sep=None, engine='python')
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

columns = {0.05: 'D_0.05(t)', 0.1: 'D_0.1(t)', 0.15: 'D_0.15(t)', 
           0.2: 'D_0.2(t)', 0.25: 'D_0.25(t)', 0.3: 'D_0.3(t)'}

concs_list = []
k1_list = []
fit_start_time = 30 
df_fit = df[df['Time_Sec'] >= fit_start_time]


plot_data = []

for conc, col in sorted(columns.items()):
    if col in df.columns:
        t = df['Time_Sec']
        D = df[col]
        
        t_fit = df_fit['Time_Sec']
        valid_mask = df_fit[col] > 0
        t_fit_v = t_fit[valid_mask]
        ln_D_v = np.log(df_fit[col][valid_mask])
        
        res = linregress(t_fit_v, ln_D_v)
        k1 = -res.slope
        
        concs_list.append(conc)
        k1_list.append(k1)
        plot_data.append((conc, t, D, k1))

plt.figure(figsize=(10, 6))
for conc, t, D, k1 in plot_data:
    t_segment = t[t >= fit_start_time]
    D_segment = D[t >= fit_start_time]
    plt.plot(t_segment, D_segment, label=f"{conc} M")
plt.xlim(left=0, right=410)
plt.xlabel("Время t, сек", fontsize=12)
plt.ylabel("Оптическая плотность D(t), отн.ед", fontsize=12)
plt.legend(title="Концентрация NaOH", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('1_Kinetics_D_t.png', dpi=300, bbox_inches='tight') 
plt.show()

plt.figure(figsize=(10, 6))
for conc, t, D, k1 in plot_data:
    t_segment = t[t >= fit_start_time]
    ln_D_segment = np.log(D[t >= fit_start_time])
    plt.plot(t_segment, ln_D_segment, label=f"{conc} M \n" r"$\ln D(t) = \ln D_0$" " - $k_1$"f"(={k1:.4f})t")
plt.xlabel("Время t, сек", fontsize=12)
plt.ylabel("ln D", fontsize=12)
plt.legend(title="Константы скорости k1", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('2_Linearization_lnD.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
ln_concs = np.log(concs_list)
ln_k1 = np.log(k1_list)
res_n = linregress(ln_concs, ln_k1)
n_value = res_n.slope
n_error = res_n.stderr 
k2_value = np.exp(res_n.intercept)
intercept_error = res_n.intercept_stderr 
k2_error = k2_value * intercept_error
plt.scatter(ln_concs, ln_k1, color='red', edgecolor='black', s=80, label='Эксперимент')
plt.plot(ln_concs, res_n.intercept + res_n.slope * ln_concs, 'b--', 
         label=f'n = {n_value:.3f} ± {n_error:.3f}')
plt.xlabel("ln [OH-]")
plt.ylabel("ln k1")
plt.legend()
plt.grid(True)
plt.savefig('3_Order_Determination.png', dpi=300)
plt.show()