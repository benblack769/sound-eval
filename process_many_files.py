import subprocess
import os

def get_all_paths(base_folder, extension, filter_fn=None):
    find_call = ["find", ".", "-name", "*."+extension, "-type", "f"]
    file_list = subprocess.check_output(find_call,cwd=base_folder).decode('utf-8').strip().split()
    filtered_files = file_list if filter_fn is None else [fname for fname in file_list if filter_fn(os.path.join(base_folder,fname))]
    return filtered_files
