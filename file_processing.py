import soundfile as sf
import numpy as np
import shutil
import subprocess
import io
import tempfile

def wav_to_raw_data(filename, samplerate):
    if not shutil.which('sox'):
        raise RuntimeError('''
        You need to install the command line tool `sox` to run this.
        On Ubuntu, this can be installed with `sudo apt-get install sox`
        ''')
    with tempfile.NamedTemporaryFile(suffix=".wav") as resampled_file:
        sox_args = [
            "sox",
            filename,
            "-r", str(samplerate),
            resampled_file.name
        ]
        try:
            subprocess.check_output(sox_args)
        except subprocess.CalledProcessError:
            with open("log/failed_file_loads.txt",'a') as logfile:
                logfile.write("process error on {} with sample rate {}\n".format(filename,samplerate))
            return None

        sig, out_samplerate = sf.read(resampled_file.name)

    assert out_samplerate == samplerate
    sig_float = sig.astype(np.float32)
    sig_vec = sig_float.sum(axis=1) / sig_float.shape[1] if len(sig_float.shape) > 1 else sig_float
    return sig_vec

def mp3_to_raw_data(filename, samplerate):
    '''
    parameters: - filename of mp3 file
                - samplerate of returned value
    returns: numpy array containing raw sound at some samplerate
    '''
    # converts mp3 to wav using system ffmpeg

    if not shutil.which('ffmpeg'):
        raise RuntimeError('''
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

    try:
        raw_wav_data = subprocess.check_output(ffmpeg_args)
    except subprocess.CalledProcessError:
        with open("log/failed_file_loads.txt",'a') as logfile:
            logfile.write("process error on {} with sample rate {}\n".format(filename,samplerate))
        return None


    sig, out_samplerate = sf.read(io.BytesIO(raw_wav_data))
    #print("sample rates!!!")
    #print(out_samplerate)
    #print(samplerate)
    assert out_samplerate == samplerate
    sig_float = sig.astype(np.float32)
    sig_vec = sig_float.sum(axis=1) / sig_float.shape[1] if len(sig_float.shape) > 1 else sig_float
    return sig_vec

#print(mp3_to_raw_data('../fma_small/000/000002.mp3',16000)[5000:5010])

def raw_data_to_wav(filename,data,samplerate):
    sf.write(filename,data,samplerate)

#def raw_data_to_mp3(filename,data,samplerate):
#    wav_file = tempfile.NamedTemporaryFile()
#    sf.write(wav_file.name,data,samplerate)
