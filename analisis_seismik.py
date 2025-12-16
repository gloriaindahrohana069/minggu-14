# %%
# analisis_seismik.py
# Analisis Data Mikroseismik Area Pengeboran



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Membaca & Inspeksi Data

print("=== MEMUAT DATA ===")

file_data = "data_mikroseismik.csv"

if not os.path.exists(file_data):
    raise FileNotFoundError(
        f"File '{file_data}' tidak ditemukan.\n"
        f"Pastikan file CSV berada di folder yang sama dengan script."
    )

df = pd.read_csv(file_data)

print("\n--- INFO DATA ---")
print(df.info())

print("\n--- STATISTIK DESKRIPTIF ---")
print(df.describe())


# 2. Pembersihan Data (Data Cleaning)

print("\n=== PEMBERSIHAN DATA ===")

# Hapus magnitudo tidak wajar (<= 0)
jumlah_awal = len(df)
df = df[df["Magnitudo"] > 0]
print(f"Data dihapus (magnitudo tidak wajar): {jumlah_awal - len(df)}")

# Cek nilai hilang
print("\n--- CEK NILAI HILANG ---")
print(df.isnull().sum())

# Hapus baris dengan NaN
df = df.dropna()
print(f"Jumlah data setelah cleaning: {len(df)}")


# 3. Konversi Tipe Data

print("\n=== KONVERSI TIPE DATA ===")

df["Waktu_UTC"] = pd.to_datetime(df["Waktu_UTC"], errors="coerce")
df = df.dropna(subset=["Waktu_UTC"])

print("Kolom Waktu_UTC berhasil dikonversi ke datetime")


# 4. Feature Engineering: Jarak Horizontal

print("\n=== PERHITUNGAN JARAK HORIZONTAL ===")

df["Jarak_Horizontal_m"] = np.sqrt(df["X_m"]*2 + df["Y_m"]*2)

print("Kolom Jarak_Horizontal_m berhasil dibuat")


# 5. Analisis Statistik & Kriteria

print("\n=== ANALISIS STATISTIK ===")

# Magnitudo terbesar & terkecil
gempa_max = df.loc[df["Magnitudo"].idxmax()]
gempa_min = df.loc[df["Magnitudo"].idxmin()]

print("\n--- GEMPA MAGNITUDO TERBESAR ---")
print(gempa_max)

print("\n--- GEMPA MAGNITUDO TERKECIL ---")
print(gempa_min)

# Kedalaman rata-rata
kedalaman_rata2 = df["Z_m"].mean()
print(f"\nKedalaman rata-rata gempa: {kedalaman_rata2:.2f} m")

# Distribusi kedalaman
dangkal = df[df["Z_m"] > -1600]
dalam = df[df["Z_m"] <= -1600]

print(f"Jumlah gempa dangkal (Z > -1600 m): {len(dangkal)}")
print(f"Jumlah gempa dalam (Z <= -1600 m): {len(dalam)}")

# Radius < 500 m
radius_500 = df[df["Jarak_Horizontal_m"] < 500]
print(f"Jumlah gempa dalam radius < 500 m: {len(radius_500)}")


# 6. Visualisasi Data

print("\n=== MEMBUAT GRAFIK ===")

# Grafik 1: Jarak vs Magnitudo
plt.figure()
plt.scatter(df["Jarak_Horizontal_m"], df["Magnitudo"])
plt.xlabel("Jarak Horizontal (m)")
plt.ylabel("Magnitudo")
plt.title("Jarak Horizontal vs Magnitudo")
plt.grid(True)
plt.show()

# Grafik 2: Histogram Kedalaman
plt.figure()
plt.hist(df["Z_m"], bins=20)
plt.xlabel("Kedalaman (m)")
plt.ylabel("Jumlah Kejadian")
plt.title("Distribusi Kedalaman Gempa Mikro")
plt.grid(True)
plt.show()

# Grafik 3: Magnitudo vs Waktu
plt.figure()
plt.plot(df["Waktu_UTC"], df["Magnitudo"], marker='o')
plt.xlabel("Waktu")
plt.ylabel("Magnitudo")
plt.title("Magnitudo Gempa Mikro terhadap Waktu")
plt.grid(True)
plt.show()

# Grafik 4: Sebaran X-Y
plt.figure()
plt.scatter(df["X_m"], df["Y_m"])
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Sebaran Horizontal Gempa Mikro")
plt.axhline(0)
plt.axvline(0)
plt.grid(True)
plt.show()

print("\n=== ANALISIS SELESAI ===")
# %%
