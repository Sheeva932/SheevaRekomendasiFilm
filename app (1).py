import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from difflib import get_close_matches

# Set page config
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")

# Load data
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mencari film yang cocok
def find_best_match(user_input):
    user_input = user_input.lower().strip()
    
    # Fungsi helper untuk normalisasi string (hilangkan tanda baca, spasi ekstra)
    def normalize_string(s):
        import re
        # Ganti tanda hubung, titik, dll dengan spasi, lalu hilangkan spasi berlebih
        normalized = re.sub(r'[-_\.\,\:\;]', ' ', s.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    # Normalisasi input user
    normalized_input = normalize_string(user_input)
    
    # Buat kolom sementara untuk pencarian yang sudah dinormalisasi
    df_temp = df_all.copy()
    df_temp['normalized_title'] = df_temp['title'].apply(normalize_string)
    
    # 1. Coba exact match dengan normalisasi
    exact_matches = df_temp[df_temp['normalized_title'] == normalized_input]
    if not exact_matches.empty:
        return exact_matches.iloc[0]['title'].lower()
    
    # 2. Coba partial match dengan normalisasi
    partial_matches = df_temp[df_temp['normalized_title'].str.contains(normalized_input, na=False, regex=False)]
    
    if not partial_matches.empty:
        # Prioritaskan berdasarkan:
        # 1. Yang mengandung kata kunci di awal judul
        # 2. Yang judulnya lebih pendek (biasanya lebih relevan)
        # 3. Yang lebih populer (asumsi: film dengan nama sederhana lebih populer)
        
        partial_matches = partial_matches.copy()
        partial_matches['title_length'] = partial_matches['title'].str.len()
        partial_matches['starts_with_input'] = partial_matches['normalized_title'].str.startswith(normalized_input)
        partial_matches['word_count'] = partial_matches['normalized_title'].str.split().str.len()
        
        # Urutkan prioritas: starts_with_input (desc), word count (asc), title length (asc)
        partial_matches = partial_matches.sort_values([
            'starts_with_input', 'word_count', 'title_length'
        ], ascending=[False, True, True])
        
        return partial_matches.iloc[0]['title'].lower()
    
    # 3. Coba pencarian dengan kata kunci individual
    input_words = normalized_input.split()
    if len(input_words) > 1:
        for word in input_words:
            if len(word) > 2:  # Hanya kata dengan panjang > 2 karakter
                word_matches = df_temp[df_temp['normalized_title'].str.contains(word, na=False, regex=False)]
                if not word_matches.empty:
                    # Pilih yang paling banyak mengandung kata dari input
                    word_matches = word_matches.copy()
                    word_matches['word_score'] = 0
                    
                    for input_word in input_words:
                        word_matches['word_score'] += word_matches['normalized_title'].str.contains(input_word, na=False).astype(int)
                    
                    word_matches['title_length'] = word_matches['title'].str.len()
                    word_matches = word_matches.sort_values(['word_score', 'title_length'], ascending=[False, True])
                    
                    if word_matches.iloc[0]['word_score'] > 0:
                        return word_matches.iloc[0]['title'].lower()
    
    # 4. Jika masih tidak ketemu, gunakan difflib dengan normalisasi
    normalized_titles = df_temp['normalized_title'].tolist()
    matches = get_close_matches(normalized_input, normalized_titles, n=5, cutoff=0.6)
    
    if matches:
        # Cari title asli yang sesuai dengan normalized match
        for match in matches:
            original_title = df_temp[df_temp['normalized_title'] == match]['title'].iloc[0]
            return original_title.lower()
    
    return None
    
# Fungsi rekomendasi film
def recommend_film(title):
    corrected = find_best_match(title)
    if not corrected:
        return None, None

    # Cari index film yang dicari
    matched_films = df_all[df_all['title'].str.lower() == corrected]
    if matched_films.empty:
        return None, None
        
    idx = matched_films.index[0]
    original_title = matched_films.iloc[0]['title']
    
    # Hitung similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter film dengan similarity >= 0.1, exclude film yang sama persis
    result_indices = []
    similarities = []
    
    for film_idx, similarity in sim_scores:
        if similarity >= 0.1:  # Semua film dengan similarity >= 0.1
            result_indices.append(film_idx)
            similarities.append(similarity)
    
    # Buat DataFrame hasil
    if result_indices:
        result = df_all.iloc[result_indices][['title', 'genres', 'overview', 'director', 'cast', 'poster_url']].copy()
        result['cosine_similarity'] = similarities
        return result, original_title
    
    return None, None

# CSS Styling
st.markdown("""<style> 
/* Global Styling */
body, .stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f36 50%, #0d1117 100%);
    color: #e2e8f0;
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
    background: linear-gradient(145deg, #1e293b 0%, #111827 100%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: #e2e8f0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
    position: relative;
    overflow: visible;
}

.film-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.1);
}

.film-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f59e0b, #10b981, #3b82f6);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.film-card:hover::before {
    opacity: 1;
}

/* Film Grid Layout */
.film-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

/* Film Poster */
.film-poster {
    width: 100%;
    height: 400px;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.film-poster img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    transition: transform 0.3s ease;
}

.film-card:hover .film-poster img {
    transform: scale(1.05);
}

/* Film Title */
.film-card h4 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #fbbf24;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    letter-spacing: -0.025em;
}

/* Film Details */
.film-card p {
    font-size: 14px;
    margin: 8px 0;
    color: #cbd5e1;
    line-height: 1.5;
}

.film-card p strong {
    color: #60a5fa;
    font-weight: 600;
}

img {
    border-radius: 10px;
    height: 270px;
    object-fit: cover;
}

/* Sinopsis / Summary */
details summary {
    cursor: pointer;
    color: #3b82f6;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 16px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    transition: all 0.3s ease;
    list-style: none;
    user-select: none;
}

details summary:hover {
    background: rgba(59, 130, 246, 0.15);
    transform: translateY(-1px);
}

details[open] summary {
    background: rgba(59, 130, 246, 0.15);
    margin-bottom: 12px;
}

details p {
    padding: 12px 16px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 8px;
    border-left: 3px solid #3b82f6;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: .5rem 2rem !important;
    border: none !important;
    transition: all .3s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(34, 197, 94, .3) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(34, 197, 94, .4) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus-within,
.stMultiSelect > div > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, .2) !important;
    outline: none !important;
}

/* Header */
h1, h2, h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
}

h1 {
    background: linear-gradient(135deg, #f59e0b, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #1e293b;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* Responsive */
@media (max-width: 768px) {
    .film-poster {
        height: 300px;
    }
    .film-card {
        padding: 18px;
    }
    .film-grid {
        grid-template-columns: 1fr;
    }
}
@media (max-width: 480px) {
    .film-poster {
        height: 240px;
    }
    .film-card {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)  

# --- Header ---
st.title("🎬 Sistem Rekomendasi Film")

# --- Banner ---
st.image("banner.jpg", use_container_width=True)

# --- Input Form ---
st.subheader("Cari rekomendasi berdasarkan judul film yang kamu suka")
with st.form(key="search_form"):
    input_title = st.text_input("Masukkan judul film:")
    submit = st.form_submit_button("Cari Rekomendasi")

# --- Hasil ---
if submit and input_title.strip() == "":
    st.warning("⚠️ Masukkan judul film terlebih dahulu.")
elif submit:
    hasil, corrected = recommend_film(input_title)

    if hasil is None or hasil.empty:
        st.warning(f"❌ Film '{input_title}' tidak ditemukan dalam database.")
    else:
        st.markdown(f"## 🔍 Rekomendasi film untuk mu :")
        st.info(f"✅ Ditemukan {len(hasil)} film yang relevan")

        for i in range(0, len(hasil), 3):
            cols = st.columns(3)
            for idx, col in enumerate(cols):
                if i + idx < len(hasil):
                    film = hasil.iloc[i + idx]
                    full_overview = film['overview']
                    poster_url = film.get('poster_url', '')

                    with col:
                        if poster_url and not pd.isna(poster_url):
                            try:
                                st.image(poster_url, use_container_width=True)
                            except:
                                st.error("🖼️ Poster tidak dapat dimuat")
                        else:
                            st.markdown(f"""<div style="width:100%;height:300px;background:linear-gradient(135deg,#374151,#1f2937);border-radius:12px;display:flex;align-items:center;justify-content:center;margin-bottom:16px;border:2px dashed #6b7280;"><div style="text-align:center;color:#9ca3af;">🎬<br><small>Poster Tidak Tersedia</small></div></div>""", unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class="film-card">
                                <h4>{film['title']}</h4>
                                <p><strong>Genre:</strong> {film['genres']}</p>
                                <p><strong>Director:</strong> {film['director']}</p>
                                <p><strong>Cast:</strong> {film['cast']}</p>
                                <p><strong>Similarity:</strong> {film['cosine_similarity']:.1%}</p>
                                <details style="margin-top:10px;">
                                    <summary>📖 Sinopsis</summary>
                                    <p style="margin-top:8px; color: #cbd5e1;">{full_overview}</p>
                                </details>
                            </div>
                        """, unsafe_allow_html=True)
