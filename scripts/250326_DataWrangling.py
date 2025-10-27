import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Your main dataset
spotify_df = pd.read_csv("Data/spotify_2023_cleaned.csv")

# Billboard chart songs (cleaned list of charted songs)
billboard_df = pd.read_csv("Data/hot-100-current.csv")

# Normalize names for matching
spotify_df['song_key'] = spotify_df['track_name'].str.lower().str.replace(r"[^\w\s]", "", regex=True)
billboard_df['song_key'] = billboard_df['title'].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Create charted flag
spotify_df['charted_billboard'] = spotify_df['song_key'].isin(billboard_df['song_key']).astype(int)

from pytrends.request import TrendReq
from datetime import datetime

pytrends = TrendReq(hl='en-US', tz=360)

def get_days_to_peak(song_name, artist, release_date):
    search_term = f"{song_name} {artist}"
    start_date = pd.to_datetime(release_date)
    end_date = start_date + pd.Timedelta(days=30)
    
    pytrends.build_payload([search_term], timeframe=f"{start_date.date()} {end_date.date()}", geo='US')
    data = pytrends.interest_over_time()
    
    if not data.empty:
        peak_date = data[search_term].idxmax()
        return (peak_date - start_date).days
    else:
        return None

#spotify_df['release_date'] = pd.to_datetime(spotify_df[['released_year', 'released_month', 'released_day']])
spotify_df['release_date'] = pd.to_datetime(
    spotify_df['released_year'].astype(str) + '-' +
    spotify_df['released_month'].astype(str).str.zfill(2) + '-' +
    spotify_df['released_day'].astype(str).str.zfill(2)
)
spotify_df['days_to_peak'] = spotify_df.apply(lambda row: get_days_to_peak(row['track_name'], row['artist(s)_name'], row['release_date']), axis=1)

spotify_df.to_csv("spotify_2023_coded.csv", index=False)

