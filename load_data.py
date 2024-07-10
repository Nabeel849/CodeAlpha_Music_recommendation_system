import pandas as pd

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Add a simulated user_id column
    data['user_id'] = data.groupby('text').ngroup()

    # Drop unnecessary columns for the recommendation system
    data = data[['user_id', 'song']]

    # Add a 'rating' column, assuming each interaction is positive (rating = 1)
    data['rating'] = 1

    return data

if __name__ == "__main__":
    file_path = 'spotify_millsongdata.csv'
    data = load_and_preprocess_data(file_path)
    print(data.head())
