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
        t_fit = df_fit['Time_Sec']
        valid_mask = df_fit[col] > 0
        t_v = t_fit[valid_mask]
        ln_D_v = np.log(df_fit[col][valid_mask])
        res = linregress(t_v, ln_D_v)
        concs_list.append(conc)
        k1_list.append(-res.slope)
        plot_data.append({
            'conc': conc,
            't': df['Time_Sec'],
            'D': df[col],
            'slope': res.slope,
            'intercept': res.intercept,
            'R2': res.rvalue**2
        })

plt.figure(figsize=(11, 7))
for d in plot_data:
    t_seg = d['t'][d['t'] >= fit_start_time]
    ln_D_seg = np.log(d['D'][d['t'] >= fit_start_time])
    
    eqn = f"$\ln D = {d['slope']:.4f}t {d['intercept']:+.2f}$ ({d['conc']}M)"
    plt.plot(t_seg, ln_D_seg, label=eqn)

plt.xlabel("Время t, сек", fontsize=12)
plt.ylabel("ln D, отн.ед", fontsize=12)
plt.legend(title="Аппроксимационные уравнения", fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('2_Linearization_Equations.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
ln_concs = np.log(concs_list)
ln_k1 = np.log(k1_list)
res_n = linregress(ln_concs, ln_k1)

order_eqn = f"$\ln k_1 = {res_n.slope:.1f} \cdot \ln[OH^-] {res_n.intercept:+.1f}$"

plt.scatter(ln_concs, ln_k1, color='red', edgecolor='black', s=100, label='Экспериментальные точки')
plt.plot(ln_concs, res_n.intercept + res_n.slope * ln_concs, 'b--', 
         label=f"Аппроксимация: {order_eqn}")
plt.xlabel("ln [OH-], отн.ед", fontsize=12)
plt.ylabel("ln k1, отн.ед", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('3_Order_Determination_Equation.png', dpi=300)
plt.show()
