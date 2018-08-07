import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import yaml
import argparse
import multiprocessing
import process_many_files
import shutil

DEBUG = False

def calc_mp3_spec_batch(source_filenames,dest_filenames, config):
    call_cmnd = ["python",
        "spectrify.py",
        ",".join(source_filenames),
        ",".join(dest_filenames),
        "--mel-bins={}".format(config['NUM_MEL_BINS']),
        "--samplerate={}".format(config['SAMPLERATE']),
        "--frame-len={}".format(config['TIME_SEGMENT_SIZE'])
    ]
    print(" ".join(call_cmnd))
    subprocess.check_call(call_cmnd)

def batch_filenames(abs_filenames):
    MP3_BATCH_SIZE = 100
    full_size = len(abs_filenames)
    #print(list(range(0,full_size+MP3_BATCH_SIZE,MP3_BATCH_SIZE)))
    res = [abs_filenames[i:min(i+MP3_BATCH_SIZE,full_size)] for i in range(0,full_size,MP3_BATCH_SIZE)]
    return res

def make_dirs(paths):
    for path in paths:
        pathdir = os.path.dirname(path)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

def process_audio_folder(source_folder, dest_folder, config):
    all_filenames = process_many_files.get_all_paths(source_folder,"mp3")

    source_abs_filenames = [os.path.join(source_folder,filename) for filename in all_filenames]
    dest_abs_filenames = [os.path.join(dest_folder,filename) for filename in all_filenames]
    make_dirs(dest_abs_filenames)

    if DEBUG:
        for src,dest in zip(batch_filenames(source_abs_filenames),batch_filenames(dest_abs_filenames)):
            calc_mp3_spec_batch(src,dest,config)
    else:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
            source_batch = batch_filenames(source_abs_filenames)
            dest_batch = batch_filenames(dest_abs_filenames)
            pool.map(calc_mp3_spec_batch, source_batch, dest_batch, [config]*len(source_batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .mp3 files into spectrogram numpy vectors")
    parser.add_argument('mp3_dataset', help='Path to folder full of .mp3 files (looks recursively into subfolders for .mp3 files).')
    parser.add_argument('output_folder', help='Path to output folder where .npy files will be stored.')
    parser.add_argument('--config', dest='config_yaml', default="default_config.yaml",
                    help='define the .yaml config file (default is "default_config.yaml")')

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_yaml))

    if os.path.isdir(args.output_folder):
        shutil.rmtree(args.output_folder)

    process_audio_folder(args.mp3_dataset,args.output_folder,config)
