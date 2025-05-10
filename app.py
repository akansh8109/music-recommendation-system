import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    data = {
        'title': ['Let It Be', 'Bohemian Rhapsody', 'Imagine', 'One', 'Hey Jude'],
        'artist': ['The Beatles', 'Queen', 'John Lennon', 'U2', 'The Beatles'],
        'genre': ['Rock', 'Rock', 'Pop', 'Rock', 'Pop']
    }
    return pd.DataFrame(data)
  
def build_model(df):
    df['combined'] = df['title'] + ' ' + df['artist'] + ' ' + df['genre']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    return tfidf_matrix

def recommend(song_title, df, tfidf_matrix):
    if song_title not in df['title'].values:
        return None
    index = df[df['title'] == song_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-4:-1][::-1]
    return df.iloc[similar_indices][['title', 'artist']]


def main():
    st.title("🎵 Music Recommendation System")
    df = load_data()
    tfidf_matrix = build_model(df)

    song = st.selectbox("Select a song", df['title'].values)

    if st.button("Get Recommendations"):
        result = recommend(song, df, tfidf_matrix)
        if result is not None:
            st.subheader("Recommended Songs:")
            st.table(result)
        else:
            st.error("Song not found.")

if __name__ == "__main__":
    main()
