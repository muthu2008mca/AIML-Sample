
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.title("🎬 K-Means Based Movie Recommender")

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv')
    books = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv')
    df = ratings.merge(books[['book_id', 'title']], on='book_id')
    df.columns = ['userId', 'movieId', 'rating', 'title']
    return df

df = load_data()
user_movie = df.pivot_table(index='userId', columns='title', values='rating')
imputer = SimpleImputer(strategy='mean')
user_movie_imputed = pd.DataFrame(imputer.fit_transform(user_movie),
                                   columns=user_movie.columns,
                                   index=user_movie.index)
scaler = StandardScaler()
user_movie_scaled = scaler.fit_transform(user_movie_imputed)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(user_movie_scaled)
user_movie_imputed['Cluster'] = clusters

def recommend_movies(user_id, n=5):
    if user_id not in user_movie_imputed.index:
        return []
    user_cluster = user_movie_imputed.loc[user_id, 'Cluster']
    similar_users = user_movie_imputed[user_movie_imputed['Cluster'] == user_cluster].drop('Cluster', axis=1)
    mean_ratings = similar_users.mean().sort_values(ascending=False)
    seen = user_movie.loc[user_id].dropna().index
    recommendations = mean_ratings.drop(seen, errors='ignore').head(n)
    return recommendations

st.sidebar.subheader("Select a User ID")
user_ids = user_movie_imputed.index.tolist()
selected_user = st.sidebar.selectbox("User ID", user_ids)

if st.sidebar.button("Recommend"):
    with st.spinner("Finding movies..."):
        recs = recommend_movies(selected_user)
        if not recs.empty:
            st.success("Top Recommendations:")
            st.write(recs)
        else:
            st.warning("No recommendations available.")


st.subheader("📊 Visualize Clusters with PCA")

pca = PCA(n_components=2)
user_2d = pca.fit_transform(user_movie_scaled)
cluster_df = pd.DataFrame({
    'PCA1': user_2d[:, 0],
    'PCA2': user_2d[:, 1],
    'Cluster': clusters
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=cluster_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80, ax=ax)
ax.set_title("Customer Segments (PCA)")
st.pyplot(fig)

st.subheader("🧠 Top Books by Cluster")
for cluster_num in sorted(user_movie_imputed['Cluster'].unique()):
    users = user_movie_imputed[user_movie_imputed['Cluster'] == cluster_num].drop(columns='Cluster')
    top_books = users.mean().sort_values(ascending=False).head(5)
    st.markdown(f"**Cluster {cluster_num}:**")
    for book, score in top_books.items():
        st.write(f"{book} (avg rating: {score:.2f})")
