import subprocess

def get_all_paths(base_folder, extension):
    find_call = ["find", ".", "-name", "*."+extension, "-type", "f"]
    file_list = subprocess.check_output(find_call,cwd=base_folder).decode('utf-8').strip().split()
    return file_list
