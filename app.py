import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ==================== Konfigurasi Halaman ====================
st.set_page_config(page_title="🎥 Rekomendasi Anime", layout="wide")

# ==================== Load Data ====================
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating", "members"])
    df = df[df["rating"] >= 1]
    df = df.drop_duplicates(subset=["name", "genre"])
    df = df.reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()
    return df

anime_df = load_data()

# ==================== Build Model ====================
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix, tfidf

knn_model, tfidf_matrix, tfidf_vectorizer = build_model(anime_df)

# ==================== Session State ====================
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== CSS Styling ====================
st.markdown("""
<style>
body, .main, .stApp {
    background-color: #F5F5F5 !important;
    color: #000000 !important;
    font-family: 'Segoe UI', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #E0E0E0 !important;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

h1, h2, h3, h4, h5, h6, label, span, .stTextInput label {
    color: #000000 !important;
}

input, textarea {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #B0BEC5 !important;
    border-radius: 8px;
    padding: 10px;
}

.stSelectbox > div, .css-1uccc91-singleValue, .css-1dimb5e {
    color: #000000 !important;
    background-color: #FFFFFF !important;
}

div[role="radiogroup"] > div > label {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    padding: 5px 10px;
    border-radius: 8px;
}

.anime-card {
    background-color: #FFFFFF;
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #5DADE2;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.anime-header {
    font-size: 18px;
    font-weight: bold;
    color: #000000 !important;
}
.anime-body {
    font-size: 14px;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== Sidebar ====================
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi", "📂 Genre"])

# ==================== HOME ====================
if page == "🏠 Home":
    st.title("🎌 Rekomendasi Anime Favorit")
    st.markdown("""
Selamat datang di website **Rekomendasi Anime Favorit**! 🎉

Website ini dirancang untuk membantu kamu menemukan anime baru yang mirip dengan yang kamu suka.

### ⚙️ Teknologi yang Digunakan:
- 🧠 Content-Based Filtering
- 📊 TF-IDF (Term Frequency–Inverse Document Frequency)
- 👥 K-Nearest Neighbors (KNN)
- 💻 Streamlit (Antarmuka pengguna)
- 🐍 Pandas & Scikit-learn (Pemrosesan data & machine learning)

### ✨ Fitur:
- Rekomendasi berdasarkan judul anime yang kamu masukkan
- Eksplorasi anime berdasarkan genre
- Riwayat pencarian dan hasil rekomendasi tersimpan
""")

    st.subheader("🔥 Top 10 Anime Paling Populer")
    top_members = anime_df.sort_values(by="members", ascending=False).head(10)
    for i in range(0, len(top_members), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top_members):
                anime = top_members.iloc[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 Genre: {anime['genre']}<br>
                            ⭐ Rating: {anime['rating']}<br>
                            👥 Members: {anime['members']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.subheader("🏆 Top 10 Anime dengan Rating Tertinggi")
    top_rating = anime_df.sort_values(by="rating", ascending=False).head(10)
    for i in range(0, len(top_rating), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top_rating):
                anime = top_rating.iloc[i + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 Genre: {anime['genre']}<br>
                            ⭐ Rating: {anime['rating']}<br>
                            👥 Members: {anime['members']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.subheader("🕘 Riwayat Pencarian")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"🔎 {item}")
    else:
        st.info("Belum ada pencarian.")

    st.subheader("🎯 Rekomendasi Terakhir")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations[-5:]):
            st.markdown(f"**📌 Dari**: {item['query']}")
            for anime in item['results']:
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-header">{anime['name']}</div>
                    <div class="anime-body">
                        📚 {anime['genre']}<br>
                        ⭐ {anime['rating']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Belum ada rekomendasi.")

# ==================== REKOMENDASI ====================
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime")
    input_text = st.text_input("🎬 Masukkan sebagian judul anime")
    type_options = ["Semua"] + sorted(anime_df["type"].dropna().unique())
    selected_type = st.selectbox("🎞️ Pilih Type Anime", type_options)

    if input_text:
        matches = anime_df[anime_df["name_lower"].str.contains(input_text.lower())]
        if not matches.empty:
            selected_title = st.selectbox("🔽 Pilih judul", matches["name"].unique())
            anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
            anime_genre = anime_row["genre"]

            st.markdown(f"📚 **Genre**: {anime_genre}  |  ⭐ **Rating**: {anime_row['rating']}")

            query_vec = tfidf_vectorizer.transform([anime_genre])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=200)

            st.success(f"🎯 Rekomendasi berdasarkan genre dari: {selected_title}")
            results, names_seen, shown = [], set(), 0

            for i in indices[0]:
                result = anime_df.iloc[i]
                name = result["name"]
                genre = result["genre"]
                a_type = result["type"]

                if name == selected_title or name in names_seen:
                    continue

                if selected_type != "Semua" and a_type != selected_type:
                    continue

                genre_input = set([g.strip().lower() for g in anime_genre.split(",")])
                genre_result = set([g.strip().lower() for g in genre.split(",")])
                common_genres = genre_input.intersection(genre_result)

                if len(common_genres) >= 1:
                    st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-header">{name}</div>
                        <div class="anime-body">
                            📚 Genre: {genre}<br>
                            ⭐ Rating: {result['rating']}<br>
                            🎞️ Type: {a_type}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    results.append({
                        "name": name,
                        "genre": genre,
                        "rating": result["rating"],
                        "type": a_type
                    })
                    names_seen.add(name)
                    shown += 1

                if shown == 5:
                    break

            if shown < 5:
                st.warning(f"Hanya ditemukan {shown} anime dengan kemiripan genre minimal 1.")

            st.session_state.history.append(f"{selected_title} (Type: {selected_type})")
            st.session_state.recommendations.append({
                "query": f"{selected_title} (Type: {selected_type})",
                "results": results
            })
        else:
            st.warning("Judul tidak ditemukan.")

# ==================== GENRE ====================
elif page == "📂 Genre":
    st.title("📂 Eksplorasi Berdasarkan Genre")

    all_genres = sorted(set(g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")))
    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)
    sort_by = st.selectbox("📊 Urutkan berdasarkan:", ["Rating", "Members"])

    if selected_genre:
        filtered_df = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]

        if not filtered_df.empty:
            query_vec = tfidf_vectorizer.transform([selected_genre])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=50)

            results = []
            for i in indices[0]:
                anime = anime_df.iloc[i]
                if selected_genre.lower() in anime["genre"].lower():
                    results.append(anime)

            if sort_by == "Rating":
                results = sorted(results, key=lambda x: x["rating"], reverse=True)
            else:
                results = sorted(results, key=lambda x: x["members"], reverse=True)

            results = results[:5]

            st.subheader(f"🎯 Rekomendasi Anime Genre '{selected_genre}'")
            for anime in results:
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-header">{anime['name']}</div>
                    <div class="anime-body">
                        📚 Genre: {anime['genre']}<br>
                        ⭐ Rating: {anime['rating']}<br>
                        👥 Members: {anime['members']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.session_state.history.append(f"Genre: {selected_genre}")
            st.session_state.recommendations.append({
                "query": f"Genre: {selected_genre}",
                "results": results
            })
        else:
            st.warning("Tidak ada anime ditemukan dengan genre tersebut.")
