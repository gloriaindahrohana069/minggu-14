# %%
# Program Tebak Angka
angka_rahasia = 17

while True:
    tebakan = int(input("Masukkan tebakan Anda: "))

    if tebakan < angka_rahasia:
        print("Terlalu kecil")
    elif tebakan > angka_rahasia:
        print("Terlalu besar")
    else:
        print("Selamat! Tebakan Anda benar")
        break

# %%
# Program Cetak Bilangan Genap
for angka in range(1, 21):
    if angka % 2 == 0:
        print(angka)

# %%
