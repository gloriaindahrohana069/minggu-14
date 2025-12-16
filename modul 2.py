# %%
# Membuat list berisi 10 bilangan
bilangan = [12, 5, 8, 20, 3, 15, 7, 10, 18, 1]

# Menambahkan satu bilangan baru
bilangan.append(25)

# Menghapus satu bilangan tertentu (misal 8)
bilangan.remove(8)

# Menampilkan bilangan terbesar dan terkecil
print("List bilangan:", bilangan)
print("Bilangan terbesar:", max(bilangan))
print("Bilangan terkecil:", min(bilangan))

# %%
# List nama mahasiswa
mahasiswa = ["Beni", "Riskha", "Arka", "Dhinda", "Nikenya", "Sunaff", "Borneo"]
# Mengurutkan secara alfabetis
mahasiswa.sort()
print("Daftar mahasiswa terurut:", mahasiswa)

# %%
import math

A = (10, 20, 30)
x, y, z = A

jarak = math.sqrt(x**2 + y**2 + z**2)
print("Jarak ke titik asal:", jarak)

# %%
tanggal_lahir = (20, 6, 2006)
def cetak_tanggal(tgl):
    print(f"{tgl[0]:02d}/{tgl[1]:02d}/{tgl[2]}")
cetak_tanggal(tanggal_lahir)

# %%
ganjil = {"Matematika", "Fisika", "Kimia"}
genap = {"Fisika", "Pemrograman", "Geologi"}

print("Hanya semester ganjil:", ganjil - genap)
print("Irisan mata kuliah:", ganjil & genap)

# %%
nilai = {
    "Riskha": [80, 85, 90],
    "Atika": [95, 75, 80],
    "Cindy": [90, 95, 100]
}
for nama, n in nilai.items():
    rata = sum(n) / len(n)
    print(nama, "rata-rata:", rata)

# %%
data = [1, 2, 2, 3, 4, 4, 5]
unik = set(data)
print("Data tanpa duplikasi:", unik)

# %%
mahasiswa = {
    "M1": {"nim": "123", "nama": "Borneo", "ipk": 3.60},
    "M2": {"nim": "124", "nama": "Beni", "ipk": 3.50},
    "M3": {"nim": "125", "nama": "Sunaff", "ipk": 3.90},
    "M4": {"nim": "125", "nama": "Arka", "ipk": 3.80}
}

ipk_tertinggi = max(mahasiswa.values(), key=lambda x: x["ipk"])
print("Mahasiswa IPK tertinggi:", ipk_tertinggi)

# %%
mahasiswa = [
    {"nama": "Theresia", "nim": "124120070", "jurusan": "Geofisika"},
    {"nama": "Caesa", "nim": "124150003", "jurusan": "Industri"},
    {"nama": "Kanawa", "nim": "124149015", "jurusan": "Rekayasa Instrumentasi dan Automasi"}
]
cari_nim = "124"
for mhs in mahasiswa:
    if mhs["nim"] == cari_nim:
        print("Mahasiswa ditemukan:", mhs)

# %%
jurusan_cari = "Geofisika"
for mhs in mahasiswa:
    if mhs["jurusan"] == jurusan_cari:
        print(mhs["nama"])

# %%
survey = {
    "Survey_A": [
        {"trace_id": 1, "receiver": "R1", "offset": 100, "QC_flag": "OK"},
        {"trace_id": 2, "receiver": "R2", "offset": 150, "QC_flag": "OK"},
        {"trace_id": 3, "receiver": "R3", "offset": 200, "QC_flag": "BAD"}
    ],
    "Survey_B": [
        {"trace_id": 4, "receiver": "R1", "offset": 120, "QC_flag": "OK"},
        {"trace_id": 5, "receiver": "R4", "offset": 250, "QC_flag": "OK"},
        {"trace_id": 6, "receiver": "R2", "offset": 300, "QC_flag": "BAD"}
    ]
}

# Trace BAD
for s in survey.values():
    for t in s:
        if t["QC_flag"] == "BAD":
            print("Trace BAD:", t["trace_id"])

# Total trace
total_trace = sum(len(s) for s in survey.values())
print("Total trace:", total_trace)

# Survey terbanyak
terbanyak = max(survey, key=lambda x: len(survey[x]))
print("Survey terbanyak:", terbanyak)

# Receiver unik
receiver = {t["receiver"] for s in survey.values() for t in s}
print("Receiver unik:", receiver)

# Frekuensi receiver
freq = {}
for s in survey.values():
    for t in s:
        freq[t["receiver"]] = freq.get(t["receiver"], 0) + 1

print("Frekuensi receiver:", freq)

# %%
sensor_data = {
    "Sensor_A": [(1, 0.12), (2, 0.15), (3, 0.30)],
    "Sensor_B": [(1, 0.20), (2, 0.18), (3, 0.25)],
    "Sensor_C": [(1, 0.11), (2, 0.10), (3, 0.40)]
}
def cari_amplitudo_maks(data):
    return {s: max(a for _, a in v) for s, v in data.items()}
def waktu_amplitudo_maks(data):
    hasil = {}
    for s, v in data.items():
        waktu, amp = max(v, key=lambda x: x[1])
        hasil[s] = waktu
    return hasil
laporan = []
amp = cari_amplitudo_maks(sensor_data)
waktu = waktu_amplitudo_maks(sensor_data)

for s in sensor_data:
    laporan.append({
        "sensor": s,
        "amplitudo_maks": amp[s],
        "waktu": waktu[s]
    })
print(laporan)

# %%
def qc_seismik(data, ambang):
    laporan = {}
    for s, isi in data.items():
        ok = bad = 0
        for t in isi["seismik"]:
            if t["offset"] > ambang:
                t["QC_flag"] = "BAD"
            if t["QC_flag"] == "OK":
                ok += 1
            else:
                bad += 1
        laporan[s] = {"OK": ok, "BAD": bad}
    return laporan

# %%
