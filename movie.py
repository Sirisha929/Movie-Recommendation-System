import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Data
ratings_dict = {
    "item": [1, 1, 1, 2, 2, 3],
    "user": ['A', 'B', 'C', 'A', 'B', 'C'],
    "rating": [5, 4, 3, 4, 2, 5]
}
df = pd.DataFrame(ratings_dict)

# Load
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Model
model = SVD()
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
accuracy.rmse(predictions)

# Recommend for User A
user_rated = df[df['user'] == 'A']['item'].tolist()
all_items = df['item'].unique()
unrated_items = [item for item in all_items if item not in user_rated]

recommendations = []
for item in unrated_items:
    pred = model.predict('A', item)
    recommendations.append((item, pred.est))

recommendations.sort(key=lambda x: x[1], reverse=True)
print("Recommendations for A:")
for movie, score in recommendations:
    print(f"Movie {movie} => {round(score, 2)}")
