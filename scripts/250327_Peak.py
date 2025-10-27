import time
from pytrends.request import TrendReq
import pandas as pd

# Create a copy of just the first 20 rows
sample_df = spotify_df.head(10).copy()

pytrends = TrendReq(hl='en-US', tz=360)

def get_days_to_peak(song_name, artist, release_date):
    try:
        search_term = f"{song_name} {artist}"
        start_date = pd.to_datetime(release_date)
        end_date = start_date + pd.Timedelta(days=30)

        # Format timeframe as required
        timeframe = f"{start_date.date()} {end_date.date()}"

        # Send request
        pytrends.build_payload([search_term], timeframe=timeframe, geo='US')
        data = pytrends.interest_over_time()

        if not data.empty:
            peak_date = data[search_term].idxmax()
            return (peak_date - start_date).days
        else:
            return None
    except Exception as e:
        print(f"Error for '{song_name}' by {artist}: {e}")
        return None

# Create empty list to store results
days_to_peak_list = []

# Loop with delay
for idx, row in sample_df.iterrows():
    days = get_days_to_peak(row['track_name'], row['artist(s)_name'], row['release_date'])
    days_to_peak_list.append(days)
    time.sleep(30)  # Delay to avoid 429 error

# Add to DataFrame
sample_df['days_to_peak'] = days_to_peak_list
