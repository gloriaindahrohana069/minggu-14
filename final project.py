# %% 
import streamlit as st
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

API_KEY = "b0634f918d71f32afc9b65855dfc0f46"

# ==========================
#   CLASS OOP
# ==========================

class WeatherApp:
    def __init__(self, api_key):
        self.api_key = api_key

    def search_city(self, city):
        """Ambil data cuaca utama berdasarkan nama kota."""
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        data = requests.get(url).json()

        return {
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"],
            "temp": data["main"]["temp"],
            "pressure": data["main"]["pressure"],
            "humidity": data["main"]["humidity"],
            "windspeed": data["wind"]["speed"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"],
        }

    def predict_weather(self, lat, lon):
        """Ambil data prediksi cuaca 5 hari dari koordinat."""
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        return requests.get(url).json()

    def convert_time(self, dt):
        """Konversi timestamp ke datetime normal."""
        return datetime.utcfromtimestamp(dt)


# ==========================
#   APLIKASI STREAMLIT
# ==========================

app = WeatherApp(API_KEY)

st.title("🌦 Weather Analysis App")

city = st.text_input("Masukkan nama kota:")

if st.button("Search"):

    # ==================== DATA CURRENT WEATHER ======================

    data = app.search_city(city)

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown("<p class='metric-label'>Weather</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{data['description']}</p>", unsafe_allow_html=True)

    with c2:
        st.markdown("<p class='metric-label'>Icon</p>", unsafe_allow_html=True)
        st.image(f"http://openweathermap.org/img/wn/{data['icon']}@2x.png")

    with c3:
        st.markdown("<p class='metric-label'>Temperature</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{data['temp']}°C</p>", unsafe_allow_html=True)

    with c4:
        st.markdown("<p class='metric-label'>Pressure</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{data['pressure']} hPa</p>", unsafe_allow_html=True)

    with c5:
        st.markdown("<p class='metric-label'>Humidity</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{data['humidity']}%</p>", unsafe_allow_html=True)

    with c6:
        st.markdown("<p class='metric-label'>Wind Speed</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{data['windspeed']} m/s</p>", unsafe_allow_html=True)


    # ==================== PREDIKSI CUACA ======================
    st.header("🔮 Weather Forecast (5 Days)")
    forecast = app.predict_weather(data["lat"], data["lon"])

    jumlah_data = st.slider(
        "Tampilkan berapa titik prediksi?",
        min_value=10,
        max_value=len(forecast["list"]),
        value=20
    )

    # ==================== TABEL FORECAST =====================
    table_data = []

    for item in forecast["list"][:jumlah_data]:
        waktu = app.convert_time(item["dt"])
        suhu = item["main"]["temp"]
        icon = item["weather"][0]["icon"]
        desc = item["weather"][0]["description"]
        humidity = item["main"]["humidity"]
        wind = item["wind"]["speed"]


        table_data.append({
            "Waktu": waktu,
            "Suhu (°C)": suhu,
            "Deskripsi": desc,
            "Icon": f"<img src='http://openweathermap.org/img/wn/{icon}.png' width='40'>"
        })

    df_table = pd.DataFrame(table_data)

    st.markdown("### 5 days weather forecast")
    st.write(df_table.to_html(escape=False), unsafe_allow_html=True)


    # ==================== ANALISIS NUMPY ======================
    temps = [row["Suhu (°C)"] for row in table_data]
    avg_temp = np.mean(temps)

    st.success(f"🌡 Rata-rata suhu berdasarkan NumPy: **{avg_temp:.2f}°C**")


    # ==================== VISUALISASI ======================
    times = [row["Waktu"] for row in table_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, temps, marker="o")
    ax.set_title("Perubahan Suhu 5 Hari")
    st.pyplot(fig)

    # ==================== FILE I/O ======================
    df_table.to_csv("hasil_prediksi_cuaca.csv", index=False)
    st.success("📁 Data berhasil disimpan ke: hasil_prediksi_cuaca.csv") 
# %%
