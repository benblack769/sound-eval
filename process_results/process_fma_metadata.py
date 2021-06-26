import pandas
import os

def format_id(id):
    return f"{id//1000:03d}/{id:06d}.mp3"

df = pandas.read_csv("data/fma_metadata/raw_tracks.csv")
ids = df['track_id']
new_dataframe = pandas.DataFrame(
    dict(
        id=[format_id(id) for id in ids],
        album_title=df['album_title'],
        artist_name=df['artist_name'],
        artist_website=df['artist_website'],
        license_title=df['license_title'],
        track_title=df['track_title'],
        track_date_created=df['track_date_created'],
    )
)
all_fnames = []
for root, dirs, files in os.walk("data/fma_small/"):
    for file in files:
        all_fnames.append(str(root[-3:]+"/"+file))

fma_small = pandas.DataFrame({
    "id": all_fnames
})

new_dataframe = pandas.merge(new_dataframe,fma_small,how="inner",on="id")

new_dataframe.to_csv("data/fma_metadata/track_info.csv",index=False)
