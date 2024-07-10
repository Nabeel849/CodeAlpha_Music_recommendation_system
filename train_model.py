from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['user_id'] = data.groupby('text').ngroup()
    data = data[['user_id', 'song']]
    data['rating'] = 1
    return data

def train_model(data):
    # Define a reader with the appropriate rating scale
    reader = Reader(rating_scale=(1, 1))

    # Load the dataset into the Surprise library
    dataset = Dataset.load_from_df(data[['user_id', 'song', 'rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Use Singular Value Decomposition (SVD) for collaborative filtering
    model = SVD()

    # Train the model
    model.fit(trainset)

    # Evaluate the model
    predictions = model.test(testset)
    print("RMSE:", accuracy.rmse(predictions))

    return model

if __name__ == "__main__":
    file_path = 'spotify_millsongdata.csv'
    data = load_and_preprocess_data(file_path)
    model = train_model(data)
