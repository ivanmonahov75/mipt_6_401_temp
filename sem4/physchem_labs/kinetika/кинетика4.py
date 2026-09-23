import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# -----------------------------
# Путь к суммарной таблице
# -----------------------------
folder = "kinetika/CData/"
file_path = os.path.join(folder, "merged2.csv")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Первый столбец — время
time_col = df.columns[0]

# Удаляем только строки без времени
df = df.dropna(subset=[time_col])

# Остальные столбцы — кинетические кривые
data_cols = list(df.columns[1:])

print("Столбец времени:", time_col)
print("Столбцы данных:", data_cols)

# -----------------------------
# Ионные силы в порядке столбцов
# -----------------------------
J_vals = np.array([0.1, 0.5, 1.0, 1.5, 1.66])

if len(J_vals) != len(data_cols):
    raise ValueError(
        f"Количество J_vals ({len(J_vals)}) не совпадает "
        f"с количеством столбцов данных ({len(data_cols)})."
    )

# -----------------------------
# Расчет k1 из ln D = ln D0 - k1 t
# -----------------------------
fit_start_time = 30
k1_vals = []

for J, col in zip(J_vals, data_cols):
    t = df[time_col]
    D = df[col]

    mask_fit = (
        (t >= fit_start_time)
        & t.notna()
        & D.notna()
        & (D > 0)
    )

    t_fit = t[mask_fit]
    ln_D_fit = np.log(D[mask_fit])

    if len(t_fit) < 2:
        print(f"{col}: слишком мало точек, пропускаю.")
        k1_vals.append(np.nan)
        continue

    res = linregress(t_fit, ln_D_fit)
    k1 = -res.slope
    k1_vals.append(k1)

    print(f"{col}: I = {J:.2f}, k1 = {k1:.6f} c^-1, R^2 = {res.rvalue**2:.4f}")

k1_vals = np.array(k1_vals)

# Оставляем только корректные значения
valid = np.isfinite(J_vals) & np.isfinite(k1_vals) & (k1_vals > 0)
J_vals = J_vals[valid]
k1_vals = k1_vals[valid]

# В статьях часто используют десятичный логарифм
log_k1 = np.log10(k1_vals)

# -----------------------------
# Параметры для 39°C
# -----------------------------
A_39 = 0.5254

# -----------------------------
# Функции ионной силы
# -----------------------------
f_DHL = np.sqrt(J_vals)
f_DHE = np.sqrt(J_vals) / (1 + np.sqrt(J_vals))
f_Davies = (np.sqrt(J_vals) / (1 + np.sqrt(J_vals))) - 0.2 * J_vals


# -----------------------------
# Функция для отрисовки и регрессии
# -----------------------------
def plot_model(x, y, label):
    res = linregress(x, y)
    zazb = res.slope / (2 * A_39)

    plt.scatter(x, y, edgecolor='black', s=60)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = res.intercept + res.slope * x_line

    plt.plot(
        x_line,
        y_line,
        label=(
            f"{label}: y = {res.slope:.2f}x {res.intercept:+.2f}\n"
            f"$Z_A Z_B$ = {zazb:.2f}, $R^2$ = {res.rvalue**2:.3f}"
        )
    )

    return res


# -----------------------------
# Построение графика
# -----------------------------
plt.figure(figsize=(12, 8))

res_davies = plot_model(f_Davies, log_k1, "Davies (3-е прибл.)")
res_dhe = plot_model(f_DHE, log_k1, "DHE (2-е прибл.)")
res_dhl = plot_model(f_DHL, log_k1, "DHL (1-е прибл.)")

plt.xlabel("$f(I)$", fontsize=12)
plt.ylabel("$\log_{10} k_1$", fontsize=12)
plt.title("Сравнение моделей первичного солевого эффекта", fontsize=13)
plt.legend(fontsize=9, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(folder, "Salt_Effect_Comparison.png"), dpi=300)
plt.show()

# -----------------------------
# Итоговый вывод
# -----------------------------
print("\nИтог:")
print(f"Теоретический наклон для десятичного логарифма: 2*A*Z_A*Z_B ≈ {2 * A_39 * 2:.2f}")

print(f"Davies: slope = {res_davies.slope:.3f}, Z_A Z_B = {res_davies.slope / (2*A_39):.3f}")
print(f"DHE:    slope = {res_dhe.slope:.3f}, Z_A Z_B = {res_dhe.slope / (2*A_39):.3f}")
print(f"DHL:    slope = {res_dhl.slope:.3f}, Z_A Z_B = {res_dhl.slope / (2*A_39):.3f}")