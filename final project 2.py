# %% 
import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# === CONFIG / API KEY ===
# =========================
API_KEY = "b0634f918d71f32afc9b65855dfc0f46"


# =========================
# ======  MODEL LAYER  =====
# =========================
class WeatherAPI:
    """Class untuk koneksi API ke OpenWeather."""

    def __init__(self, api_key):
        self.api_key = api_key

    def get_current_weather(self, city):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        return requests.get(url).json()

    def get_forecast(self, lat, lon):
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        return requests.get(url).json()

    def get_air_quality(self, lat, lon):
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={self.api_key}"
        return requests.get(url).json()


# =========================
# ===== UTILITY CLASS =====
# =========================
class WeatherUtils:

    @staticmethod
    def rekomendasi(kondisi, suhu):
        kondisi = (kondisi or "").lower()

        if "rain" in kondisi:
            return "🌧 Cuaca hujan — disarankan membawa payung atau jas hujan."
        elif "clear" in kondisi:
            if suhu is not None and suhu >= 33:
                return "☀ Cerah & panas — gunakan sunscreen dan minum cukup air."
            else:
                return "☀ Cerah — cuaca baik."
        elif "cloud" in kondisi:
            return "☁ Mendung — boleh membawa jaket tipis."
        elif "storm" in kondisi or "thunder" in kondisi:
            return "⛈ Badai — hindari aktivitas di luar ruangan."
        elif "wind" in kondisi:
            return "💨 Berangin — hati-hati berkendara."

        if suhu is not None:
            if suhu >= 33:
                return "🥵 Panas — minum lebih banyak."
            elif suhu <= 20:
                return "🥶 Dingin — gunakan jaket tebal."

        return "ℹ Cuaca normal — tidak ada rekomendasi khusus."

    @staticmethod
    def ts_to_dt(ts):
        try:
            return datetime.utcfromtimestamp(int(ts))
        except:
            return None


# =========================
# ===== UI LAYER ==========
# =========================
class WeatherApp:

    def __init__(self):
        self.api = WeatherAPI(API_KEY)

    # --------------------
    # Plotly Chart Helper
    # --------------------
    def line_chart(self, title, x, y, y_label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(width=3)))

        fig.update_layout(
            title=title,
            title_x=0.0,
            xaxis_title="",
            yaxis_title=y_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey")
        )
        return fig

    # --------------------
    # HEADER
    # --------------------
    def header(self):
        st.set_page_config(page_title="Aplikasi Cuaca", layout="wide")
        st.markdown("<h1 style='text-align:center'>🌤 Aplikasi Cuaca</h1>", unsafe_allow_html=True)

        dark = st.checkbox("🌙 Aktifkan Dark Mode", key="dark_mode")
        if dark:
            st.markdown("""
                <style>
                    .stApp { background:#0b0f12; color:#e6eef6; }
                </style>
            """, unsafe_allow_html=True)

    # --------------------
    # INPUT
    # --------------------
    def input_city(self):
        col, _ = st.columns([3, 1])
        with col:
            city = st.text_input("Masukkan nama kota:", value="Jakarta")
            run = st.button("Cari Cuaca")

        if run and not city.strip():
            st.error("Masukkan nama kota terlebih dahulu.")
            return None, False

        return city, run

    # --------------------
    # CURRENT WEATHER
    # --------------------
    def show_current_weather(self, data, city):
        st.markdown("## 🌍 Informasi Cuaca Saat Ini")
        desc = data["weather"][0]["description"]
        icon = data["weather"][0].get("icon")
        main = data["main"]
        wind = data["wind"]
        coord = data["coord"]

        col = st.columns([2, 1, 1, 1, 1])

        with col[0]:
            st.subheader(city.title())
            if icon:
                st.image(f"http://openweathermap.org/img/wn/{icon}@2x.png", width=100)
            st.write(f"**Deskripsi:** {desc}")
            st.write(f"**Suhu:** {main['temp']} °C")
            st.write(f"**Kelembapan:** {main['humidity']} %")
            st.write(f"**Tekanan:** {main.get('pressure')} hPa")
            st.write(f"**Angin:** {wind.get('speed')} m/s")

        col[1].metric("Suhu (°C)", main["temp"])
        col[2].metric("Kelembapan (%)", main["humidity"])
        col[3].metric("Angin (m/s)", wind["speed"])
        col[4].metric("Koordinat", f"{coord['lat']:.4f}, {coord['lon']:.4f}")

        st.markdown("### 📝 Rekomendasi Aktivitas")
        st.info(WeatherUtils.rekomendasi(desc, main["temp"]))

        return coord["lat"], coord["lon"]

    # --------------------
    # AIR QUALITY
    # --------------------
    def show_air_quality(self, aq):
        st.markdown("## 🌫 Air Quality (AQI)")

        if "list" not in aq or len(aq["list"]) == 0:
            st.write("Tidak ada data AQI.")
            return

        item = aq["list"][0]
        aqi = item["main"]["aqi"]
        comp = item["components"]

        aqi_text = {1: "Baik 😄", 2: "Cukup Baik 🙂", 3: "Sedang 😐", 4: "Buruk 😷", 5: "Sangat Buruk 🤢"}
        st.metric("AQI", f"{aqi} — {aqi_text.get(aqi)}")

        df = pd.DataFrame([comp]).T.rename(columns={0: "µg/m³"})
        st.dataframe(df)

    # --------------------
    # FORECAST
    # --------------------
    def show_forecast(self, forecast, lat, lon, city):
        st.markdown("## 📅 Prediksi Cuaca (5 Hari — 3 Jam)")

        if str(forecast.get("cod")) != "200":
            st.write("Forecast tidak tersedia.")
            return

        items = forecast["list"]
        max_pts = len(items)
        pts = st.slider("Jumlah titik prediksi:", 8, max_pts, min(24, max_pts))

        rows = []
        for it in items[:pts]:
            rows.append({
                "Waktu": WeatherUtils.ts_to_dt(it["dt"]),
                "Suhu (°C)": it["main"]["temp"],
                "Kelembapan (%)": it["main"]["humidity"],
                "Angin (m/s)": it["wind"]["speed"],
                "Deskripsi": it["weather"][0]["description"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv().encode(), f"forecast_{city}.csv")

        st.markdown("### 🌡 Grafik Suhu")
        st.plotly_chart(self.line_chart("Hourly Temperature", df["Waktu"], df["Suhu (°C)"], "°C"), use_container_width=True)

        st.markdown("### 💧 Grafik Kelembapan")
        st.plotly_chart(self.line_chart("Hourly Humidity", df["Waktu"], df["Kelembapan (%)"], "%"), use_container_width=True)

        st.markdown("### 🌬 Kecepatan Angin")
        st.plotly_chart(self.line_chart("Hourly Wind Speed", df["Waktu"], df["Angin (m/s)"], "m/s"), use_container_width=True)

    # --------------------
    # MAIN RUN
    # --------------------
    def run(self):
        self.header()
        city, run = self.input_city()
        if not run:
            return

        with st.spinner("Mengambil data cuaca..."):
            data = self.api.get_current_weather(city)

            if str(data.get("cod")) != "200":
                st.error(f"Kota tidak ditemukan! ({data.get('message')})")
                return

            lat, lon = self.show_current_weather(data, city)

            st.markdown("## 🗺 Lokasi Kota")
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)

            aq = self.api.get_air_quality(lat, lon)
            self.show_air_quality(aq)

            forecast = self.api.get_forecast(lat, lon)
            self.show_forecast(forecast, lat, lon, city)

            st.success("✅ Data berhasil diambil.")


# =========================
# ====== RUN APP ==========
# =========================
if __name__ == "__main__":
    WeatherApp().run()

# %%
