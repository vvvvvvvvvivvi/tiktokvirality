import time
import pandas as pd

# Your main dataset
spotify_df = pd.read_csv("Data/spotify_2023_peak.csv")

# Convert both columns to datetime (if they aren't already)
spotify_df['release_date'] = pd.to_datetime(spotify_df['release_date'])
spotify_df['peak_day'] = pd.to_datetime(spotify_df['peak_day'])

# Calculate days_to_peak as the difference
spotify_df['days_to_peak'] = (spotify_df['peak_day'] - spotify_df['release_date']).dt.days

# Set all NaN and negative values to 0
spotify_df['days_to_peak'] = spotify_df['days_to_peak'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)

spotify_df.to_csv("spotify_2023_peak_cleaned.csv", index=False)

