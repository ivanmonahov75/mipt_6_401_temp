import os
import glob
import re
import numpy as np
import pandas as pd


folder = "kinetika/CData/"


def natural_key(path):
    """
    Нормальная сортировка:
    IV2_1.csv, IV2_2.csv, ..., IV2_10.csv
    а не IV2_1.csv, IV2_10.csv, IV2_2.csv
    """
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


# Берем все файлы IV*.csv из папки
files = sorted(
    glob.glob(os.path.join(folder, "IV*.csv")),
    key=natural_key
)

print("Найденные файлы:")
for file in files:
    print(os.path.basename(file))


def read_iv_file(path):
    name = os.path.splitext(os.path.basename(path))[0]

    df = pd.read_csv(path)

    # В этих файлах время оказалось в индексе,
    # поэтому возвращаем индекс в обычный столбец
    df = df.reset_index()

    # Берем первые два столбца: время и оптическую плотность
    df = df.iloc[:, :2].copy()
    df.columns = ["Time_Sec", name]

    # Заменяем десятичные запятые на точки
    df["Time_Sec"] = df["Time_Sec"].astype(str).str.replace(",", ".", regex=False)
    df[name] = df[name].astype(str).str.replace(",", ".", regex=False)

    # Переводим в числа
    df["Time_Sec"] = pd.to_numeric(df["Time_Sec"], errors="coerce")
    df[name] = pd.to_numeric(df[name], errors="coerce")

    # Убираем строки с мусором
    df = df.dropna(subset=["Time_Sec", name])

    return df


tables2 = [read_iv_file(file) for file in files]

if len(tables2) == 0:
    raise ValueError("Не найдено файлов IV*.csv в указанной папке.")

merged2 = tables2[0]

for table in tables2[1:]:
    merged2 = pd.merge(
        merged2,
        table,
        on="Time_Sec",
        how="outer"
    )

merged2 = merged2.sort_values("Time_Sec").reset_index(drop=True)

# numpy-массивы
time2_np = merged2["Time_Sec"].to_numpy()
values2_np = merged2.drop(columns=["Time_Sec"]).to_numpy()
data2_np = merged2.to_numpy()

print("\nИтоговая таблица merged2:")
print(merged2.head())

print("\nСтолбцы merged2:")
print(list(merged2.columns))

print("\nРазмер data2_np:", data2_np.shape)
print("Размер time2_np:", time2_np.shape)
print("Размер values2_np:", values2_np.shape)

# Сохранение итоговой таблицы
merged2.to_csv(os.path.join(folder, "merged2.csv"), index=False)

np.save(os.path.join(folder, "merged2.npy"), data2_np)
np.save(os.path.join(folder, "merged2_time.npy"), time2_np)
np.save(os.path.join(folder, "merged2_values.npy"), values2_np)import os
import numpy as np
import pandas as pd


folder = "kinetika/CData/"

# Уже собранный файл по вторым данным
merged2_path = os.path.join(folder, "merged2.csv")

# Читаем готовую суммарную таблицу
merged2 = pd.read_csv(merged2_path)

# Чистим названия столбцов
merged2.columns = merged2.columns.str.strip()

# Переводим все данные в числа
for col in merged2.columns:
    merged2[col] = pd.to_numeric(merged2[col], errors="coerce")

# Первый столбец считаем временем
time_col = merged2.columns[0]

# Переименуем первый столбец в Time_Sec, если он называется иначе
if time_col != "Time_Sec":
    merged2 = merged2.rename(columns={time_col: "Time_Sec"})

# Убираем только строки без времени
merged2 = merged2.dropna(subset=["Time_Sec"])

# Сортируем по времени
merged2 = merged2.sort_values("Time_Sec").reset_index(drop=True)

# numpy-массивы
time2_np = merged2["Time_Sec"].to_numpy()
values2_np = merged2.drop(columns=["Time_Sec"]).to_numpy()
data2_np = merged2.to_numpy()

print("\nИтоговая таблица merged2:")
print(merged2.head())

print("\nСтолбцы merged2:")
print(list(merged2.columns))

print("\nРазмер data2_np:", data2_np.shape)
print("Размер time2_np:", time2_np.shape)
print("Размер values2_np:", values2_np.shape)

# Сохраняем обратно очищенную версию, если нужно
merged2.to_csv(os.path.join(folder, "merged2_clean.csv"), index=False)

np.save(os.path.join(folder, "merged2.npy"), data2_np)
np.save(os.path.join(folder, "merged2_time.npy"), time2_np)
np.save(os.path.join(folder, "merged2_values.npy"), values2_np)