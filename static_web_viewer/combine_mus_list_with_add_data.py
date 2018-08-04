import pandas
import numpy as np

def read_file(filename):
    with open(filename) as file:
        return file.read()

def read_lines(filename):
    return read_file(filename).strip().split("\n")


music_file_list = read_lines("../example_vectors/music_list.txt")
music_mid_list = [filename[:-4] for filename in music_file_list]
add_dataframe = pandas.read_csv("../../../phil_projs/midiworld_download/view_csv.csv")
print((add_dataframe['filename'].dtype))
filename_mapping = {name:genre for name,genre in zip(add_dataframe.filename,add_dataframe.genre)}
genre_list = [filename_mapping[name] for name in music_mid_list]
combined_data = pandas.DataFrame({
    "genre":genre_list,
    "filename": music_mid_list
})
print(combined_data.shape)
combined_data.to_csv("../example_vectors/annotated_data.csv",index=False)
