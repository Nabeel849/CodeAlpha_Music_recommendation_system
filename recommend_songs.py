from train_model import train_model, load_and_preprocess_data
import pandas as pd

def get_top_n_recommendations(model, data, user_id, n=10):
    # Get all song IDs
    all_song_ids = data['song'].unique()

    # Predict ratings for all songs for the specified user
    predictions = [model.predict(user_id, song_id) for song_id in all_song_ids]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_n_recommendations = predictions[:n]

    return [(pred.iid, pred.est) for pred in top_n_recommendations]

if __name__ == "__main__":
    file_path = 'spotify_millsongdata.csv'
    data = load_and_preprocess_data(file_path)
    model = train_model(data)

    user_id = 1  # Replace with the user ID you want to generate recommendations for
    recommendations = get_top_n_recommendations(model, data, user_id, n=10)

    print("Top 10 song recommendations for user {}:".format(user_id))
    for song_id, estimated_rating in recommendations:
        print("Song: {}, Estimated Rating: {:.2f}".format(song_id, estimated_rating))
