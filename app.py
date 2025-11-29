import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import urllib.parse

# ==================== Page Config ====================
st.set_page_config(page_title="🎥 Rekomendasi Anime — Sleek UI", layout="wide")

# ==================== Helper: Image URL from Unsplash ====================
def unsplash_image_url(query: str, w: int = 900, h: int = 506):
    q = urllib.parse.quote_plus(query)
    return f"https://source.unsplash.com/{w}x{h}/?{q},anime"

# ==================== Load Data ====================
@st.cache_data
def load_data(path="anime.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["name", "genre", "rating", "members"])
    df = df[df["rating"] >= 0]
    df = df.drop_duplicates(subset=["name", "genre"]).reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()
    if "type" not in df.columns:
        df["type"] = "TV"
    df["image"] = df["name"].apply(lambda x: unsplash_image_url(x))
    return df

anime_df = load_data()

# ==================== Build Model ====================
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df["genre"].fillna(df["name"])
    tfidf_matrix = tfidf.fit_transform(corpus)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix, tfidf

knn_model, tfidf_matrix, tfidf_vectorizer = build_model(anime_df)

# ==================== Custom Global CSS ====================
st.markdown("""
<style>
:root { --bg1: #1b0003; --bg2: #350006; --card: #3b0b0f; --accent: #ff4667; }
body, .stApp { background: linear-gradient(180deg,var(--bg1),var(--bg2)); color: #fff; font-family: 'Poppins', sans-serif; }

/* NAVBAR FIXED TOP */
.navbar{ position:fixed; top:0; left:0; right:0; height:70px; z-index:9999; display:flex; align-items:center; justify-content:space-between; padding:0 40px; background:#070405; box-shadow:0 4px 20px rgba(0,0,0,0.6); }
.nav-left{ display:flex; align-items:center; gap:16px; }
.logo{ font-weight:700; font-size:22px; }
.nav-menu{ display:flex; gap:22px; align-items:center; }
.nav-menu a{ color:#ccc; text-decoration:none; padding:10px 16px; border-radius:8px; }
.nav-menu a.active, .nav-menu a:hover{ background: rgba(255,255,255,0.06); color:#fff; }

/* Add top margin for content */
.content-wrap{ margin-top:90px; }

/* Hero */
.hero{ width:100%; height:480px; border-radius:12px; overflow:hidden; position:relative; display:flex; align-items:center; }
.hero .bg{ position:absolute; inset:0; background-size:cover; background-position:center; filter:brightness(0.45); }
.hero .content{ position:relative; padding:50px; max-width:60%; }

/* Cards */
.carousel{ display:flex; gap:18px; overflow-x:auto; padding-bottom:12px; }
.card{ min-width:230px; background:var(--card); border-radius:12px; overflow:hidden; box-shadow:0 8px 20px rgba(0,0,0,0.6); }
.card img{ width:100%; height:150px; object-fit:cover; }
.card .meta{ padding:12px; }

</style>
""", unsafe_allow_html=True)

# ==================== Navbar ====================
st.markdown("""
<div class='navbar'>
  <div class='nav-left'>
    <div class='logo'>🎬 <span style='color:var(--accent)'>Movie</span>Anime</div>
  </div>
  <div class='nav-menu'>
    <a href='/?page=home' class='nav-item'>Home</a>
    <a href='/?page=rekom' class='nav-item'>Rekomendasi</a>
    <a href='/?page=genre' class='nav-item'>Genre</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== Page Detection via Query Params ====================
params = st.query_params
page = params.get("page", "home")

st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

# ==================== HOME ====================
if page == "home":
    hero_img = anime_df.iloc[0]["image"]

    st.markdown(f"""
    <div class='hero'>
      <div class='bg' style="background-image: url('{hero_img}');"></div>
      <div class='content'>
        <h1>Selamat Datang di MovieAnime</h1>
        <p>Temukan anime terbaik, trending, dan rekomendasi personal.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔥 Trending Anime")
    st.markdown("<div class='carousel'>", unsafe_allow_html=True)
    for _, row in anime_df.sort_values(by='members', ascending=False).head(10).iterrows():
        st.markdown(f"""
        <div class='card'>
          <img src='{row['image']}' />
          <div class='meta'>
            <h4>{row['name']}</h4>
            <p>{row['genre']}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== REKOMENDASI ====================
elif page == "rekom":
    st.title("🔎 Rekomendasi Anime")
    input_text = st.text_input("Masukkan judul anime:")

    if input_text:
        matches = anime_df[anime_df["name_lower"].str.contains(input_text.lower())]

        if not matches.empty:
            selected_title = st.selectbox("Pilih Judul", matches["name"].unique())

            anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
            genre_text = anime_row["genre"]

            query_vec = tfidf_vectorizer.transform([genre_text])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=20)

            st.subheader(f"Rekomendasi mirip: {selected_title}")
            st.markdown("<div class='carousel'>", unsafe_allow_html=True)

            for i in indices[0]:
                result = anime_df.iloc[i]
                if result["name"] == selected_title:
                    continue
                st.markdown(f"""
                <div class='card'>
                  <img src='{result['image']}' />
                  <div class='meta'>
                    <h4>{result['name']}</h4>
                    <p>⭐ {result['rating']}</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Judul tidak ditemukan.")

# ==================== GENRE ====================
elif page == "genre":
    st.title("📂 Eksplorasi Genre")
    all_genres = sorted(set(g for genres in anime_df["genre"] for g in genres.split(",")))
    selected_genre = st.selectbox("Pilih Genre", all_genres)

    filtered = anime_df[anime_df["genre"].str.contains(selected_genre, case=False)]

    st.subheader(f"Genre: {selected_genre}")
    st.markdown("<div class='carousel'>", unsafe_allow_html=True)
    for _, row in filtered.head(12).iterrows():
        st.markdown(f"""
        <div class='card'>
          <img src='{row['image']}' />
          <div class='meta'>
            <h4>{row['name']}</h4>
            <p>⭐ {row['rating']}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
