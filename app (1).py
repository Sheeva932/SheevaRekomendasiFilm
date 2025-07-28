
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
    /* Global Styling */
    body, .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f36 50%, #0d1117 100%);
        color: #e6edf3;
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Film Card Styling */
    .film-card {
        background: linear-gradient(145deg, #1c2333 0%, #252b42 100%);
        border: 1px solid rgba(101, 109, 118, 0.2);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        color: #e6edf3;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .film-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #f97316, #eab308, #22c55e, #3b82f6, #8b5cf6);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .film-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        border-color: rgba(101, 109, 118, 0.4);
    }
    
    .film-card:hover::before {
        opacity: 1;
    }
    
    /* Film Title */
    .film-card h4 {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f97316;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.025em;
    }
    
    /* Film Details */
    .film-card p {
        font-size: 14px;
        margin: 8px 0;
        color: #c9d1d9;
        line-height: 1.5;
    }
    
    .film-card p strong {
        color: #58a6ff;
        font-weight: 600;
    }

    img {
        border-radius: 10px;
        height: 270px;
        object-fit: cover;
    }

    /* Rating Badge */
    .rating-badge {
        display: inline-block;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
    }
    
    /* Details/Summary */
    details {
        margin-top: 16px;
    }
    
    details summary {
        cursor: pointer;
        color: #58a6ff;
        font-size: 14px;
        font-weight: 500;
        padding: 8px 16px;
        background: rgba(88, 166, 255, 0.1);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 8px;
        transition: all 0.3s ease;
        list-style: none;
        user-select: none;
    }
    
    details summary:hover {
        background: rgba(88, 166, 255, 0.15);
        border-color: rgba(88, 166, 255, 0.3);
        transform: translateY(-1px);
    }
    
    details[open] summary {
        background: rgba(88, 166, 255, 0.15);
        border-color: rgba(88, 166, 255, 0.3);
        margin-bottom: 12px;
    }
    
    details p {
        padding: 12px 16px;
        background: rgba(13, 17, 23, 0.5);
        border-radius: 8px;
        border-left: 3px solid #58a6ff;
        margin: 0;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within,
    .stMultiSelect > div > div > div:focus-within {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
        outline: none !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #238636) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(35, 134, 54, 0.4) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #161b22 !important;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #f97316, #eab308);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem !important;
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .loading-card {
        background: linear-gradient(90deg, #1c2333 25%, #252b42 50%, #1c2333 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .film-card {
            padding: 18px;
            margin-bottom: 16px;
        }
        
        .film-card h4 {
            font-size: 18px;
        }
        
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #161b22;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #30363d, #21262d);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #484f58, #30363d);
    }
    </style>

""", unsafe_allow_html=True)

# --- Banner ---
st.image("banner.jpg", use_container_width=True)

# --- Input ---
st.subheader("Cari rekomendasi berdasarkan judul film yang kamu suka")

# Form agar bisa jalan pakai Enter juga
with st.form(key="search_form"):
    input_title = st.text_input("Masukkan judul film:")
    submit = st.form_submit_button("Cari Rekomendasi")

# Jalankan hasil jika submit atau Enter
if submit and input_title.strip() == "":
    st.warning("⚠️ Masukkan judul film terlebih dahulu.")
elif submit and input_title:
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
