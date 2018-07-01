import pandas
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import csv
import sklearn.linear_model
import numpy as np
import collections


#def get_overall_dataframe()
metadata_folder = "../fma_metadata/"
tracks_file = metadata_folder+"raw_tracks.csv"
genres_file = metadata_folder+"genres.csv"

def get_random_genre_id(genre_map,genre_str):
    genre_items = json.loads(genre_str.replace("'",'"'))
    #print(genre_csv.genre_id == genre_items[0]['genre_id'])
    genre_parents = [genre_map[gen['genre_id']] for gen in genre_items]
    unique_items = list(set(genre_parents))
    genre_choice = random.choice(unique_items)
    return genre_choice

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

def get_test_train(data1,data2):
    train_keys = list(np.random.choice(np.arange(0,len(data1)),size=int(0.5*len(data1))))
    train_key_set = set(train_keys)
    test_keys = [x for x in range(0,len(data1)) if x not in train_key_set]

    train_data1 = [data1[key] for key in train_keys]
    train_data2 = [data2[key] for key in train_keys]
    test_data1 = [data1[key] for key in test_keys]
    test_data2 = [data2[key] for key in test_keys]
    return (train_data1,train_data2),(test_data1,test_data2)

def run_regression(high_d_data,genre_data):
    genre_boolean_data = [gen == '17'for gen in genre_data]
    print(sum(genre_boolean_data)/len(genre_boolean_data))
    logit_model = sklearn.linear_model.LogisticRegression()
    (train_highd,train_genre),(test_highd,test_genre) = get_test_train(high_d_data,genre_boolean_data)
    logit_model.fit(train_highd,train_genre)
    prediction = logit_model.predict(test_highd)
    score = logit_model.score(test_highd,test_genre)

    print(score)
    print(sum(prediction))

    #arg = pandas.DataFrame(high_d_data)
    #arg['genre_id'] = genre_data
    #print(arg)
    #logit_model.fit(high_d_data,)
    #print(collections.Counter(genre_data))

def get_genre_ids(music_list_file):
    tracks_csv = pandas.read_csv(tracks_file)
    genre_csv = list(csv.reader(open(genres_file)))
    genre_map = {gen[0]:gen[4] for gen in genre_csv}
    tracks_filtered = filter_tracks(tracks_csv, music_list_file)

    genre_list = tracks_filtered['track_genres'].tolist()
    genre_ids = [get_random_genre_id(genre_map,gen) for gen in genre_list]
    return genre_ids


def associate_metadata(data_2d, genre_ids):
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
    raw_data = "../relu_repo_results/vector_at_4110.npy"
    plot_tsne_results = "process_results/relu_repo_results_4110_2d.npy"
    full_d_data = np.load(raw_data)
    tsne_data = np.load(plot_tsne_results)
    genre_ids = get_genre_ids(music_filename)
    run_regression(full_d_data,genre_ids)
    #associate_metatdata = associate_metadata(tsne_data,genre_ids)
    #plot_associated_data(associate_metatdata)
    print("finisehed")
