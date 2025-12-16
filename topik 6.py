# %%
def buat_salam(nama):
    return f"Selamat pagi, {nama}!"

# Memanggil fungsi
hasil = buat_salam("Gloria Indah Rohana Lumbantoruan")
print(hasil)

# %% 
#Perluas Library Kalkulator

# kalkulator_lib.py
def tambah(a, b):
    return a + b
def kurang(a, b):
    return a - b
def kali(a, b):
    return a * b
def bagi(a, b):
    if b == 0:
        return "Error: pembagian dengan nol!"
    return a/b

# tambahkan fungsi sesuai tugas
def pangkat(angka, pangkat):
    return angka ** pangkat

# main py
def tampilkan_menu():
    print("/n=== MENU KALKULATOR ===")
    print("1. Tambah")
    print("2. Kurang")
    print("3. Kali")
    print("4. Bagi")
    print("5. Pangkat")
    print("0. Keluar")

while True:
    tampilkan_menu()
    pilihan = input("Pilih menu: ")

    if pilihan == "0":
        print("Keluar dari program.")
        break
    elif pilihan in ["1", "2", "3", "4", "5"]:
        a = float(input("Masukkan angka pertama: "))

        # kalau fungsi pangkat, input kedua adalah pangkat
        if pilihan == "5":
            b = float(input("Masukkan pangkat: "))
            print("Hasil:",pangkat(a, b))

        else:
            b = float(input("Masukkan angka kedua: "))
            if pilihan == "1":
                print("Hasil:", tambah(a, b))
            elif pilihan == "2":
                print("Hasil:", kurang(a, b))
            elif pilihan == "3":
                print("Hasil:", kali(a, b))
            elif pilihan == "4":
                print("Hasil:", bagi(a, b))

    else:
        print("Menu tidak valid!")
# %%
