import pandas as pd

movies = pd.read_csv('C:/Users/nlers/Documents/315code/315private/pa2/movie-lens-data/movie-lens-data/movies.csv')
links = pd.read_csv('C:/Users/nlers/Documents/315code/315private/pa2/movie-lens-data/movie-lens-data/links.csv')
ratings = pd.read_csv('C:/Users/nlers/Documents/315code/315private/pa2/movie-lens-data/movie-lens-data/ratings.csv')
tags = pd.read_csv('C:/Users/nlers/Documents/315code/315private/pa2/movie-lens-data/movie-lens-data/tags.csv')

df = pd.merge(ratings, movies, on='movieId')

movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
corr_matrix = movie_matrix.corr(method = 'pearson', min_periods=5)

with open("output.txt", 'w') as dataout:
    for i in range(1,len(movie_matrix)):
        user_ratings=movie_matrix.iloc[i].dropna() 
        recommended=pd.Series()
        for j in range(0,len(user_ratings)):
            similar = corr_matrix[user_ratings.index[j]].dropna()
            similar = similar.map(lambda x: x * user_ratings[j])
        recommended = recommended.append(similar)
        recommended.sort_values(inplace=True, ascending=False)
        dataout.write('\n' + str(i) + '\n')
        dataout.write(recommended.head(5).to_string())
