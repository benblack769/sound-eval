# sound-eval

This repository contains command line tools to process audio files (in mp3 or wav formats) into vector embeddings.


### Installing

System libraries: ffmpeg, sox

    sudo apt-get install ffmpeg
    sudo apt-get install sox # only needed to process wav files

Python libraries: numpy, soundfile, tensorflow, matplotlib pyyaml

    pip install numpy soundfile tensorflow matplotlib pyyaml

Make sure to install tensorflow-gpu if you want to use a gpu.

Note that some of the post-processing tools have dependencies of their own that also need to be installed if you wish to use them.

### High level usage information

It takes a few steps to processing raw files into something useful. A high level description of those steps is here. API details are in the files themselves, and can be seen by calling the files directly with no arguments or with a -h.


#### Preprocessing

Preprocess the audio files into lists of [mel scale](https://en.wikipedia.org/wiki/Mel_scale) frequency vectors. This is done with the `process_many_files.py` tool, which converts a nested folder of sound files (wav or mp3) into the same nested structure, but with .npy files (which contain a list of FFT vectors).

example:

    python spectrify_all.py example_input/ example_vectors/ --config=configs/small_train.yaml

#### Training

Training is done by the `spectrogram_doc2vec.py` file. This file takes in the folder of vectors produced by the preprocessing, and trains the model on those vectors. Note that training operates on an infinite loop with no cutoff, so make sure to cancel the operation when loss as printed starts to flatten. The output folder saves the state of training so training can be easily restarted by rerunning the command with the same output folder.

    python spectrogram_doc2vec.py example_vectors/ example_outs/ --config=configs/small_train.yaml

#### Output folder details

The output folder has a lot of files. You mostly need to know about the first two files in this list to use all the tools. Most of the tools pick out these files automatically, but a few don't for increased flexibility.

* `vector_at_{x}.npy`, is the song level vectors saved at iteration `num`
* `music_list.txt` stores the filenames of the vectors in the same order as the vectors are stores in `vector_at_{x}.npy`. This is the **only** reliable way of matching up vectors to their original document.
* `weights` folder holds the wegiths for the model. This is only saved for the very last iteration, so it **will not** necessarily correspond in any meaningful way to `vector_at_{x}.npy` unless `num` is the very maximum (the one stored in `epoc_num.txt`).
* `epoc_num.txt` stores the last iteration number, or the number at which the `weights` were saved. Needed to correctly match up `weights` and vectors (used by `calc_all_vecs.py` for instance).  
* `config.yaml` is a copy of the configuration file that was used. Useful for reproducibility reasons.
* `cost_list.csv` stores the cost at different iterations as the model is trained. Useful for examining long runs.

### Postprocessing

The resulting vectors can be used for many tasks. There are several options for processing the output vectors already implemented.

* Constructing an interactive display for the sound file embeddings. `static_web_viewer\display_vector_data.py`
* Using supervised SVMs to classify the songs based on their vector, and a labeled training dataset: `process_results\logistic_regress.py`
* Automatically create categories of sound with spherical k-means, and compare those categories to labeled categories. `process_results\sphere_kmeans.py`

Examples of these are listed below:

### API details

## Successful Examples on the FMA dataset

The FMA dataset is https://github.com/mdeff/fma
