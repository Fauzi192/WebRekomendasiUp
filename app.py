import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import urllib.parse

# ==================== Page Config ====================
st.set_page_config(page_title="🎥 Rekomendasi Anime — Sleek UI", layout="wide")

# ==================== Helper: Image URL from Unsplash Source ====================
def unsplash_image_url(query: str, w: int = 900, h: int = 506):
    # Use Unsplash Source to fetch an image without API key. Browser will load actual image.
    # Encode query to be URL safe
    q = urllib.parse.quote_plus(query)
    return f"https://source.unsplash.com/{w}x{h}/?{q},anime"

# ==================== Load Data ====================
@st.cache_data
def load_data(path="anime.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["name", "genre", "rating", "members"]) if set(["name","genre","rating","members"]).issubset(df.columns) else df.dropna()
    # keep only reasonable ratings
    if "rating" in df.columns:
        df = df[df["rating"] >= 0]
    df = df.drop_duplicates(subset=["name", "genre"]) if set(["name","genre"]).issubset(df.columns) else df.drop_duplicates()
    df = df.reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()
    # ensure type column exists
    if "type" not in df.columns:
        df["type"] = "TV"
    # ensure image column may exist, if not we'll generate
    if "image" not in df.columns:
        df["image"] = df["name"].apply(lambda x: unsplash_image_url(x))
    return df

anime_df = load_data()

# ==================== Build Model (TF-IDF + KNN) ====================
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    # If genre column missing, fallback to name
    corpus = df["genre"].fillna(df["name"]) if "genre" in df.columns else df["name"]
    tfidf_matrix = tfidf.fit_transform(corpus)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix, tfidf

knn_model, tfidf_matrix, tfidf_vectorizer = build_model(anime_df)

# ==================== Session State ====================
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== Custom CSS (deep red theme, hero, cards, navbar) ====================
st.markdown("""
<style>
:root { --bg1: #2a0006; --bg2: #4b0009; --card: #3b0b0f; --accent: #ff5873; }
body, .main, .stApp { background: linear-gradient(180deg,var(--bg1) 0%, #2d0a0a 40%); color: #fff; font-family: 'Poppins', sans-serif; }

/* Navbar */
.navbar{ display:flex; align-items:center; justify-content:space-between; padding:18px 40px; background:#070405; box-shadow:0 2px 12px rgba(0,0,0,0.6); border-bottom: 4px solid rgba(255,255,255,0.02); }
.nav-left{ display:flex; align-items:center; gap:18px; }
.logo{ font-weight:700; font-size:20px; color:#fff; }
.nav-menu{ display:flex; gap:22px; align-items:center; }
.nav-menu a{ color: #ddd; text-decoration:none; padding:8px 14px; border-radius:8px; }
.nav-menu a.active, .nav-menu a:hover{ background: rgba(255,255,255,0.04); color:#fff; }

/* Hero */
.hero{ width:100%; height:520px; border-radius:12px; overflow:hidden; position:relative; display:flex; align-items:center; }
.hero .bg{ position:absolute; inset:0; background-size:cover; background-position:center; filter:brightness(0.45) saturate(1.2); }
.hero .content{ position:relative; z-index:2; padding:56px; max-width:60%; }
.hero h1{ font-size:48px; margin:0 0 10px 0; letter-spacing: -1px; }
.hero p{ opacity:0.9; }
.hero .cta{ margin-top:20px; display:flex; gap:12px; }
.btn{ padding:10px 18px; border-radius:10px; border:none; cursor:pointer; font-weight:600; }
.btn-primary{ background: linear-gradient(90deg,var(--accent), #ff3d57); color: #fff; }
.btn-ghost{ background: transparent; border:1px solid rgba(255,255,255,0.12); color:#fff; }

/* Section headings */
.section{ padding:28px 40px; }
.section h2{ margin:6px 0 18px 0; }
.carousel{ display:flex; gap:18px; overflow-x:auto; padding-bottom:14px; }
.card{ min-width:220px; background:var(--card); border-radius:10px; overflow:hidden; box-shadow:0 8px 20px rgba(0,0,0,0.6); }
.card img{ width:100%; height:130px; object-fit:cover; display:block; }
.card .meta{ padding:12px; }
.card .meta h4{ margin:0 0 6px 0; font-size:16px; }
.card .meta p{ margin:0; font-size:13px; opacity:0.9; }

/* Genre badges */
.badge{ display:inline-block; padding:6px 10px; background:rgba(0,0,0,0.35); border-radius:999px; margin-right:8px; font-size:12px; }

/* Footer tweak */
.footer{ padding:30px 40px; color:rgba(255,255,255,0.7); font-size:14px; }

/* small screens */
@media (max-width: 900px){ .hero .content{ max-width:100%; padding:28px; } .hero{ height:420px; } }
</style>
""", unsafe_allow_html=True)

# ==================== Navbar (rendered via markdown) ====================
st.markdown("""
<div class='navbar'>
  <div class='nav-left'>
    <div class='logo'>🎬 <span style='color:var(--accent)'>Movie</span> Anime</div>
  </div>
  <div class='nav-menu'>
    <a href='#' class='active'>Home</a>
    <a href='#'>Anime</a>
    <a href='#'>Movies</a>
    <a href='#'>Manga</a>
    <a href='#'>Blog</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== Page Selection (3 pages) ====================
page = st.radio("", ["🏠 Home", "🔎 Rekomendasi", "📂 Genre"], horizontal=True)

# ==================== HOME ====================
if page == "🏠 Home":
    # Hero
    hero_img = anime_df.iloc[0]['image'] if not anime_df.empty else unsplash_image_url('anime hero')
    st.markdown(f"""
    <div class='section'>
      <div class='hero'>
        <div class='bg' style="background-image: url('{hero_img}');"></div>
        <div class='content'>
          <div style='color:#ffd6df; font-weight:600; margin-bottom:8px;'>New Release</div>
          <h1>Naruto Clein</h1>
          <p style='max-width:540px;'>Temukan rekomendasi anime yang sesuai selera kamu. Jelajahi koleksi, lihat detil, dan dapatkan rekomendasi personal berdasarkan genre.</p>
          <div class='cta'>
            <button class='btn btn-primary'>Play Now</button>
            <button class='btn btn-ghost'>More Info</button>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Carousels: Trending, New Releases, Top Picks
    st.markdown("""
    <div class='section'>
      <h2>🔥 Trending Shows</h2>
      <div class='carousel'>
    """, unsafe_allow_html=True)

    # pick some high members as trending
    trending = anime_df.sort_values(by='members', ascending=False).head(8)
    for _, row in trending.iterrows():
        img = row['image'] if pd.notna(row['image']) else unsplash_image_url(row['name'])
        st.markdown(f"""
        <div class='card'>
          <img src='{img}' alt='cover'/>
          <div class='meta'>
            <h4>{row['name']}</h4>
            <p>{row['genre'] if 'genre' in row else ''}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Rated
    st.markdown("""
    <div class='section'>
      <h2>🏆 Top Rated</h2>
      <div class='carousel'>
    """, unsafe_allow_html=True)

    top_rating = anime_df.sort_values(by='rating', ascending=False).head(8)
    for _, row in top_rating.iterrows():
        img = row['image']
        st.markdown(f"""
        <div class='card'>
          <img src='{img}' alt='cover'/>
          <div class='meta'>
            <h4>{row['name']}</h4>
            <p>⭐ {row['rating']} • {row['type']}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer small
    st.markdown("""
    <div class='footer'>
      <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>Movie Anime © 2025 — built with ❤️</div>
        <div>Follow us: Instagram • YouTube • Twitter</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== REKOMENDASI ====================
elif page == "🔎 Rekomendasi":
    st.markdown("""
    <div class='section'>
      <h2>🔍 Cari Rekomendasi Anime</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        input_text = st.text_input("🎬 Masukkan sebagian judul anime (mis. Naruto)")
    with col2:
        type_options = ["Semua"] + sorted(anime_df["type"].dropna().unique())
        selected_type = st.selectbox("🎞️ Pilih Type", type_options)

    if input_text:
        matches = anime_df[anime_df["name_lower"].str.contains(input_text.lower())]
        if not matches.empty:
            selected_title = st.selectbox("🔽 Pilih judul", matches["name"].unique())
            anime_row = anime_df[anime_df["name"] == selected_title].iloc[0]
            anime_genre = anime_row["genre"]

            st.markdown(f"""
            <div class='section'>
              <h3>Rekomendasi untuk: <strong>{selected_title}</strong></h3>
            </div>
            """, unsafe_allow_html=True)

            query_vec = tfidf_vectorizer.transform([anime_genre])
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=200)

            results, names_seen, shown = [], set(), 0
            st.markdown("<div class='carousel'>", unsafe_allow_html=True)
            for i in indices[0]:
                result = anime_df.iloc[i]
                name = result['name']
                if name == selected_title or name in names_seen:
                    continue
                a_type = result['type']
                if selected_type != 'Semua' and a_type != selected_type:
                    continue
                # filter by at least one common genre
                genre_input = set([g.strip().lower() for g in anime_genre.split(",")])
                genre_result = set([g.strip().lower() for g in result['genre'].split(",")]) if 'genre' in result else set()
                if len(genre_input.intersection(genre_result)) >= 1:
                    img = result['image']
                    st.markdown(f"""
                    <div class='card'>
                      <img src='{img}' alt='cover'/>
                      <div class='meta'>
                        <h4>{result['name']}</h4>
                        <p>⭐ {result['rating']} • {result['type']}</p>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    results.append(result)
                    names_seen.add(name)
                    shown += 1
                if shown == 8:
                    break
            st.markdown("</div>", unsafe_allow_html=True)

            if shown == 0:
                st.warning("Tidak ditemukan rekomendasi yang cocok.")

            # save history
            st.session_state.history.append(selected_title)
            st.session_state.recommendations.append({"query": selected_title, "results": [r['name'] for r in results]})
        else:
            st.warning("Judul tidak ditemukan di dataset.")

# ==================== GENRE ====================
elif page == "📂 Genre":
    st.markdown("""
    <div class='section'>
      <h2>📂 Eksplorasi Berdasarkan Genre</h2>
    </div>
    """, unsafe_allow_html=True)

    all_genres = sorted(set(g.strip() for genres in anime_df["genre"].dropna() for g in genres.split(",")))
    selected_genre = st.selectbox("🎭 Pilih Genre", all_genres)
    sort_by = st.selectbox("📊 Urutkan berdasarkan:", ["Rating", "Members"])

    if selected_genre:
        filtered_df = anime_df[anime_df["genre"].str.contains(selected_genre, case=False, na=False)]
        if filtered_df.empty:
            st.warning("Tidak ada anime untuk genre ini.")
        else:
            # sort
            if sort_by == 'Rating':
                filtered_df = filtered_df.sort_values(by='rating', ascending=False)
            else:
                filtered_df = filtered_df.sort_values(by='members', ascending=False)

            st.markdown("<div class='carousel'>", unsafe_allow_html=True)
            for _, row in filtered_df.head(12).iterrows():
                img = row['image']
                st.markdown(f"""
                <div class='card'>
                  <img src='{img}' alt='cover'/>
                  <div class='meta'>
                    <h4>{row['name']}</h4>
                    <p>⭐ {row['rating']} • {row['type']}</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # save
            st.session_state.history.append(f"Genre: {selected_genre}")
            st.session_state.recommendations.append({"query": f"Genre: {selected_genre}", "results": list(filtered_df['name'].head(8))})

# ==================== End ====================


