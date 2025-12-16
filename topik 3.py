# %%
# Program Konversi Suhu Celcius ke Fahrenheit

celcius = float(input("Masukkan suhu dalam Celcius: "))

fahrenheit = (celcius * 9/5) + 32

print(f"Suhu {celcius}°C sama dengan {fahrenheit}°F")

# %%
# Program Cek Angka Ganjil atau Genap

angka = int(input("Masukkan sebuah bilangan bulat: "))

if angka % 2 == 0:
    print(f"Angka {angka} adalah bilangan genap")
else:
    print(f"Angka {angka} adalah bilangan ganjil")
