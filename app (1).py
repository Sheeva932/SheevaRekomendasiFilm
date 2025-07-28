
import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load file .pkl
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Fungsi Rekomendasi ---
def recommend_film(title):
    title = title.lower()
    matches = df_all[df_all['title'].str.lower().str.contains(title, na=False)]

    if matches.empty or title.strip() == "":
        return None

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Hapus film itu sendiri & filter minimal 0.1
    sim_scores = [score for score in sim_scores if score[0] != idx and score[1] >= 0.1]

    film_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]

    result = df_all.iloc[film_indices][['title', 'genres', 'overview', 'director', 'cast', 'poster_url']].copy()
    result['cosine_similarity'] = similarities
    return result

# --- CSS Tampilan ---
st.title("🎬 Sistem Rekomendasi Film")
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0c1e3c;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .film-card {
        background-color: #14253d;
        padding: 15px;
        border-radius: 15px;
        height: auto;
        color: #fff;
        margin-bottom: 15px;
    }
    .film-card h4 {
        font-size: 18px;
        margin-bottom: 8px;
        color: #ffa726;
    }
    .film-card p {
        font-size: 14px;
        margin: 3px 0;
    }
    details summary {
        cursor: pointer;
        color: #90caf9;
        font-size: 14px;
    }
    .stTextInput>div>div>input {
        background-color: #1e3a5f;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.image("banner.jpg", use_container_width=True)

# --- Input ---
st.subheader("Cari rekomendasi berdasarkan judul film yang kamu suka")
input_title = st.text_input("Masukkan judul film:")

# Jalankan saat tekan enter atau klik tombol
if st.button("Cari Rekomendasi"):
    hasil = recommend_film(input_title)
    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.markdown("### Berikut hasil rekomendasi film untukmu:")
if input_title:
    hasil = recommend_film(input_title)
    if hasil is None or hasil.empty:
        st.warning(f"❌ Film dengan judul '{input_title}' tidak ditemukan atau tidak ada yang mirip.")
    else:
        st.markdown("## 🔍 Berikut hasil rekomendasi film untuk mu : ")
        cols = st.columns(3)
        for i, (_, row) in enumerate(hasil.iterrows()):
            full_overview = row['overview']
            short_overview = full_overview[:200] + "..." if len(full_overview) > 200 else full_overview
            with cols[i % 3]:
                st.markdown(f'''
                    <div class="film-card">
                        <img src="{row['poster_url']}" width="100%" style="border-radius: 10px; margin-bottom: 10px;">
                        <h4>{row['title']}</h4>
                        <p><b>Genre:</b> {row['genres']}</p>
                        <p><b>Director:</b> {row['director']}</p>
                        <p><b>Cast:</b> {row['cast']}</p>
                        <p><b>Overview:</b> {short_overview}</p>
                        <details style="margin-top:5px;"><summary>Sinopsis</summary><p>{full_overview}</p></details>
                    </div>
                ''', unsafe_allow_html=True)
else:
    st.info("Silakan masukkan judul film terlebih dahulu.")
