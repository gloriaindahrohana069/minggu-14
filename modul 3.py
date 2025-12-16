# %%
class Mahasiswa:
    def __init__(self, nama, nim, prodi):
        self.nama = nama
        self.nim = nim
        self.prodi = prodi

    def tampilkan_data(self):
        print(f"Nama: {self.nama}, NIM: {self.nim}, Prodi: {self.prodi}")

m1 = Mahasiswa("Dhinda", "124120060", "Geofisika")
m2 = Mahasiswa("Nasywa", "124120055", "Geofisika")
m3 = Mahasiswa("Arizal", "124120052", "Geofisika")

m1.tampilkan_data()
m2.tampilkan_data()
m3.tampilkan_data()
# %%
class Sensor:
    def __init__(self, nilai):
        self.__nilai_sinyal = nilai

    def get_nilai(self):
        return self.__nilai_sinyal

    def set_nilai(self, nilai):
        if nilai >= 0:
            self.__nilai_sinyal = nilai
        else:
            print("Nilai tidak valid")

sensor1 = Sensor(20)
print(sensor1.get_nilai())
sensor1.set_nilai(69)
print(sensor1.get_nilai())
sensor1.set_nilai(-5)

# %%
class AlatGeofisika:
    def __init__(self, nama, tipe):
        self.nama = nama
        self.tipe = tipe

    def info(self):
        print(f"Nama: {self.nama}, Tipe: {self.tipe}")

class Magnetometer(AlatGeofisika):
    def __init__(self, nama, tipe, sensitivitas):
        super().__init__(nama, tipe)
        self.sensitivitas = sensitivitas

    def info(self):
        super().info()
        print(f"Sensitivitas: {self.sensitivitas}")

class Gravimeter(AlatGeofisika):
    def __init__(self, nama, tipe, resolusi):
        super().__init__(nama, tipe)
        self.resolusi = resolusi

    def info(self):
        super().info()
        print(f"Resolusi: {self.resolusi}")

alat1 = Magnetometer("Mag-01", "Magnetik", 0.01)
alat2 = Gravimeter("Grav-01", "Gravitasi", 0.001)

alat1.info()
alat2.info()

data = [1, 2, 2, 3, 4, 4]
print(set(data))

# %%
class Sensor:
    def baca_data(self):
        print("Membaca data sensor umum")

class SensorSeismik(Sensor):
    def baca_data(self):
        print("Membaca data amplitudo seismik")

class SensorMagnetik(Sensor):
    def baca_data(self):
        print("Membaca data medan magnet")

sensor_list = [Sensor(), SensorSeismik(), SensorMagnetik()]

for s in sensor_list:
    s.baca_data()

# %%
import random

class Sensor:
    def __init__(self, nama, lokasi):
        self.nama = nama
        self.lokasi = lokasi

class SensorSeismik(Sensor):
    def __init__(self, nama, lokasi):
        super().__init__(nama, lokasi)
        self.data = []

    def tambah_data(self, nilai):
        self.data.append(nilai)

    def rata_rata(self):
        return sum(self.data)/len(self.data) if self.data else 0

sensor1 = SensorSeismik("Seismo-1", "Sumatera")

for i in range(10):
    sensor1.tambah_data(random.uniform(1, 10))

print("Rata-rata amplitudo:", sensor1.rata_rata())

# %%
import random

# =========================
# KELAS DASAR
# =========================
class Sensor:
    def __init__(self, id_sensor, lokasi):
        self.id_sensor = id_sensor
        self.lokasi = lokasi
        self.status = "Nonaktif"
        self.data = []

    def aktifkan_sensor(self):
        self.status = "Aktif"

    def nonaktifkan_sensor(self):
        self.status = "Nonaktif"

    def rekam_data(self):
        if self.status == "Aktif":
            nilai = random.uniform(1, 10)
            self.data.append(nilai)
        else:
            print(f"Sensor {self.id_sensor} tidak aktif")

    def rata_rata(self):
        return sum(self.data) / len(self.data) if self.data else 0


# =========================
# KELAS TURUNAN
# =========================
class SensorSeismik(Sensor):
    def __init__(self, id_sensor, lokasi, frekuensi):
        super().__init__(id_sensor, lokasi)
        self.frekuensi = frekuensi


class SensorGravitasi(Sensor):
    def __init__(self, id_sensor, lokasi, resolusi):
        super().__init__(id_sensor, lokasi)
        self.resolusi = resolusi


# =========================
# PROGRAM UTAMA
# =========================
sensor1 = SensorSeismik("S-01", "Sumatera", 50)
sensor2 = SensorGravitasi("G-01", "Jawa", 0.01)

# List of object
daftar_sensor = [sensor1, sensor2]

# Aktifkan sensor
for sensor in daftar_sensor:
    sensor.aktifkan_sensor()

# Simulasi perekaman data
for i in range(10):
    for sensor in daftar_sensor:
        sensor.rekam_data()

# =========================
# LAPORAN SENSOR
# =========================
print("LAPORAN DATA SENSOR")
print("-" * 70)
print(f"{'ID':<10}{'Lokasi':<15}{'Status':<10}{'Jumlah Data':<15}{'Rata-rata':<10}")
print("-" * 70)

for sensor in daftar_sensor:
    print(f"{sensor.id_sensor:<10}{sensor.lokasi:<15}{sensor.status:<10}"
          f"{len(sensor.data):<15}{sensor.rata_rata():.2f}")

# %%
