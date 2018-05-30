import soundfile as sf
import numpy as np
import shutil
import subprocess
import io

def mp3_to_raw_data(filename, samplerate=16000):
    '''
    parameters: - filename of mp3 file
                - samplerate of returned value
    returns: numpy array containing raw sound at some samplerate
    '''
    # converts mp3 to wav using system ffmpeg

    if not shutil.which('ffmpeg'):
        raise runtime_error('''
        You need to install the command line tool `ffmpeg` to run this.
        On Ubuntu, this can be installed with `sudo apt-get install ffmpeg`
        ''')

    ffmpeg_args = [
        "ffmpeg",
        "-i", filename, # sets input file
        "-ar", str(int(samplerate)), # sets sample rate
        "-f", "wav",    # sets output type
        "-loglevel", "warning", # suppreses output except for warnings
        "pipe:1"    # puts output in stdout, allowing check_output to get the data
    ]
    print(" ".join(ffmpeg_args))
    raw_wav_data = subprocess.check_output(ffmpeg_args)

    sig, samplerate = sf.read(io.BytesIO(raw_wav_data))
    sig_float = sig.astype(np.float32)
    sig_vec = sig_float.sum(axis=1) / sig_float.shape[1]
    return sig_vec

#print(mp3_to_raw_data('../fma_small/000/000002.mp3',16000)[5000:5010])
