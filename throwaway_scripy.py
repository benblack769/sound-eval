import process_fma_files
import os

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

music_paths, raw_data_list = process_fma_files.get_raw_data_list(16000,2000)
print(music_paths)
save_str = "\n".join([os.path.basename(path)[:-4] for path in music_paths])
save_string("music_list.txt",save_str)
