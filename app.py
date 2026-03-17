import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# OPTIONAL SPOTIFY
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET"
    ))
    spotify_available = True
except:
    spotify_available = False

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="MusicAI Pro", page_icon="🎧", layout="wide")

# --------------------------------------------------
# UI DESIGN
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #121212, #1DB954, #0f2027, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.card {
    background: rgba(24,24,24,0.85);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 20px;
    margin-bottom: 20px;
    border-left: 5px solid #1DB954;
    transition: 0.4s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 20px #1DB954;
}
.scroll {
    animation: scrollDown 1.5s ease-in-out;
}
@keyframes scrollDown {
    from {opacity:0; transform: translateY(50px);}
    to {opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 Nasih's Music AI Pro")

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
conn = sqlite3.connect("music_system.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    username TEXT,
    song TEXT,
    mood TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# --------------------------------------------------
# AUTH
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    cursor.execute("INSERT OR IGNORE INTO users VALUES (?,?,?)",
                   (username, hash_password(password), "user"))
    conn.commit()

def authenticate(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                   (username, hash_password(password)))
    return cursor.fetchone()

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    st.subheader("🔐 Login / Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.user = username
            st.rerun()
        else:
            st.error("Invalid login")

    if st.button("Register"):
        create_user(username, password)
        st.success("User created")

    st.stop()

st.sidebar.write(f"👤 {st.session_state.user}")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("dataset.csv").dropna().drop_duplicates()
df = df.reset_index(drop=True)

# --------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# --------------------------------------------------
# SAFE FEATURE DETECTION
# --------------------------------------------------
def get_pipeline_features(pipeline):

    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    if isinstance(pipeline, dict):
        if "scaler" in pipeline and hasattr(pipeline["scaler"], "feature_names_in_"):
            return list(pipeline["scaler"].feature_names_in_)

    return [
        'danceability','energy','acousticness','valence',
        'tempo','speechiness','liveness',
        'loudness','instrumentalness','key','mode','duration_ms'
    ]

features = get_pipeline_features(pipeline)

for col in features:
    if col not in df.columns:
        df[col] = 0

# --------------------------------------------------
# SAFE TRANSFORM
# --------------------------------------------------
def transform_features(data, pipeline):

    if hasattr(pipeline, "transform"):
        return pipeline.transform(data)

    elif isinstance(pipeline, dict):
        X = data.copy()
        if "scaler" in pipeline:
            X = pipeline["scaler"].transform(X)
        if "pca" in pipeline:
            X = pipeline["pca"].transform(X)
        return X

    else:
        st.error("Invalid pipeline")
        st.stop()

X = transform_features(df[features], pipeline)

# --------------------------------------------------
# SPOTIFY PREVIEW
# --------------------------------------------------
def get_preview(track, artist):
    if not spotify_available:
        return None
    try:
        result = sp.search(q=f"{track} {artist}", limit=1)
        return result["tracks"]["items"][0]["preview_url"]
    except:
        return None

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🎧 Recommendation", "📜 History", "📊 Admin", "🤖 AI"]
)

# ==================================================
# RECOMMENDATION
# ==================================================
if page == "🎧 Recommendation":

    song = st.text_input("Search Song")

    if st.button("Generate"):

        with st.spinner("🎧 AI analyzing..."):
            time.sleep(1.5)

        match = df[df["track_name"].str.lower().str.contains(song.lower())]

        if match.empty:
            st.error("Song not found")
        else:
            idx = match.index[0]

            sim = cosine_similarity(X[idx].reshape(1,-1), X)[0]
            indices = sorted(range(len(sim)), key=lambda i: sim[i], reverse=True)[1:20]

            st.markdown('<div class="scroll">', unsafe_allow_html=True)

            for i in indices:
                track = df.iloc[i]["track_name"]
                artist = df.iloc[i]["artists"]

                preview = get_preview(track, artist)

                st.markdown(f"""
                <div class="card">
                🎵 <b>{track}</b><br>
                👤 {artist}
                </div>
                """, unsafe_allow_html=True)

                if preview:
                    st.audio(preview)

            st.markdown('</div>', unsafe_allow_html=True)

            # SAFE IMAGE DISPLAY
            st.markdown("---")
            if os.path.exists("music_banner.png"):
                st.image("music_banner.png", use_container_width=True)
            else:
                st.image(
                    "https://images.unsplash.com/photo-1511379938547-c1f69419868d",
                    use_container_width=True
                )

# ==================================================
# HISTORY
# ==================================================
elif page == "📜 History":
    logs = pd.read_sql("SELECT * FROM logs", conn)
    st.dataframe(logs)

# ==================================================
# ADMIN
# ==================================================
elif page == "📊 Admin":
    logs = pd.read_sql("SELECT * FROM logs", conn)
    if not logs.empty:
        st.bar_chart(logs["song"].value_counts())

# ==================================================
# AI ASSISTANT
# ==================================================
elif page == "🤖 AI":
    q = st.text_input("Ask")
    if st.button("Ask"):
        st.write("Music recommendation uses similarity between audio features.")