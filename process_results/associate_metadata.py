import pandas
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt


#def get_overall_dataframe()
metadata_folder = "../fma_metadata/"
tracks_file = metadata_folder+"raw_tracks.csv"

def get_random_genre_id(genre_str):
    genre_items = json.loads(genre_str.replace("'",'"'))
    genre_choice = random.choice(genre_items)
    return genre_choice['genre_id']

def get_sound_ids(music_list_file):
    track_raw_ids = open(music_list_file).readlines()
    track_ids = [int(t_id) for t_id in track_raw_ids]
    return track_ids

def filter_tracks(tracks_csv, music_list_file):
    sound_ids = get_sound_ids(music_list_file)
    sound_id_set = set(sound_ids)
    assert len(sound_ids) == len(sound_id_set)
    tracks_filtered = tracks_csv[tracks_csv.track_id.isin(sound_ids)]
    return tracks_filtered

def associate_metadata(data_2d, music_list_file):
    tracks_csv = pandas.read_csv(tracks_file)
    tracks_filtered = filter_tracks(tracks_csv, music_list_file)

    genre_list = tracks_filtered['track_genres'].tolist()
    genre_ids = [get_random_genre_id(gen) for gen in genre_list]


    xvals,yvals = np.transpose(data_2d)

    associate_dataframe = pandas.DataFrame(data={
        'x':xvals,
        'y':yvals,
        'genre_ids':genre_ids,
    })
    return associate_dataframe


def plot_associated_data(df):
    groups = df.groupby('genre_ids')

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
    ax.legend()

    plt.show()

    #dataframe.plot(x='xvals',y='yvals',kind="scatter")

    #ass_plot.show()
    #print(dataframe.tostring())
    #plt


if __name__ == "__main__":
    music_filename = "music_list.txt"
    plot_tsne_results = "process_results/relu_repo_results_999_2d.npy"
    data = np.load(plot_tsne_results)
    associate_metatdata = associate_metadata(data,music_filename)
    plot_associated_data(associate_metatdata)
    print("finisehed")
