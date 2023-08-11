from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


movie_data = pd.read_csv('movies.csv')
features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for i in features:
    movie_data[i] = movie_data[i].fillna("")

combined_feature = movie_data['genres'] + " " + movie_data['keywords'] + " " + movie_data['tagline'] + " " + movie_data['cast'] + " " + movie_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_feature)
similar = cosine_similarity(feature_vectors)
recommend_title = movie_data['title'].tolist()

@app.route("/", methods=['GET', 'POST'])
def index():
    
    movie_names = []  # Initialize the list outside the conditional block

    if request.method == 'POST':
        input_movie = request.form['user_input']
        if input_movie:
            close_match = difflib.get_close_matches(input_movie, recommend_title)
            if close_match:  # Make sure there's a close match before proceeding
                close_match = close_match[0]
                index_movie = movie_data[movie_data.title == close_match]['index'].values[0]
                similar_score = list(enumerate(similar[index_movie]))
                sorted_similar_score = sorted(similar_score, key=lambda x: x[1], reverse=True)
                i = 1
                for movie in sorted_similar_score:
                    index = movie[0]
                    title_index = movie_data[movie_data.index == index]['title'].values[0]
                    if i < 10:
                        movie_names.append(title_index)
                        i += 1

    return render_template('index.html', movie_names=movie_names)

if __name__ == '__main__':
    app.run(port=3500, debug=True)
