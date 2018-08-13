import tensorflow.contrib.signal as tfsignal
import tensorflow as tf
import numpy as np
import argparse
from file_processing import mp3_to_raw_data,wav_to_raw_data

LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 7600.0

def tf_spectrify(signals, num_mel_bins, samplerate, time_frame_length):
    num_frames = int(samplerate * time_frame_length)
    stfts = tf.contrib.signal.stft(signals, frame_length=2**11, frame_step=num_frames,
                               fft_length=2**11)
    #power_spectrograms = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, samplerate, LOWER_EDGE_HERTZ,
      UPPER_EDGE_HERTZ)
    mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for <a href="../../api_docs/python/tf/tensordot"><code>tf.tensordot</code></a> does not currently handle this case.
    # mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
    #  linear_to_mel_weight_matrix.shape[-1:]))`
    log_offset = 1e-6
    log_magnitude_spectrograms = tf.log(mel_spectrograms + log_offset)

    return log_magnitude_spectrograms


def spectrify_audios(audio_list, num_mel_bins, samplerate, time_frame_len):
    signals = tf.placeholder(tf.float32, [1, None])
    spectrogram = tf_spectrify(signals,num_mel_bins, samplerate, time_frame_len)

    config = tf.ConfigProto(
        device_count = {'GPU': int(False)}
    )
    with tf.Session(config=config) as sess:
        spectrogram_list = []
        for raw_sound in audio_list:
            if raw_sound is not None:
                pow_spec_res = sess.run([spectrogram],feed_dict={
                    signals: raw_sound.reshape((1,len(raw_sound))),
                })
                spectrogram_list.append(pow_spec_res[0][0])
            else:
                spectrogram_list.append(None)

    return spectrogram_list

def calc_mp3_spectrogram(mp3_filename, num_mel_bins, samplerate, time_frame_len):
    ext = mp3_filename.split(".")[-1]
    if ext  == "mp3":
        raw_data = mp3_to_raw_data(mp3_filename,samplerate)
    elif ext == "wav":
        raw-data = wav_to_raw_data(mp3_filename,samplerate)
    res = None if raw_data is None else  calc_spectrogram(raw_data, num_mel_bins, samplerate, time_frame_len)
    return res

def calc_spectrogram(raw_sound, num_mel_bins, samplerate, time_frame_len):
    return spectrify_audios([raw_sound],num_mel_bins,samplerate, time_frame_len)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a mp3 encoded sound file to a spectrogram")
    parser.add_argument('mp3_input_files', help='Paths to input .mp3 file')
    parser.add_argument('out_spec_files', help='Path to output .npy spectrogram file')
    parser.add_argument('--mel-bins', dest='num_mel_bins', help='number of spectrogram output bins')
    parser.add_argument('--samplerate', dest='samplerate', help='samplerate to process sound file')
    parser.add_argument('--frame-len', dest='frame_len', help='length of each spectrogram frame')

    args = parser.parse_args()

    samplerate = int(args.samplerate)
    input_fnames = args.mp3_input_files.split(",")
    output_fnames = args.out_spec_files.split(",")

    assert len(input_fnames) == len(output_fnames), "input and output lists must be same size"
    spec_datas = [mp3_to_raw_data(mp3_filename,samplerate) for mp3_filename in input_fnames]

    spec_outs = spectrify_audios(
        spec_datas,
        int(args.num_mel_bins),
        int(args.samplerate),
        float(args.frame_len),
    )
    for val, filename in zip(spec_outs,output_fnames):
        if val is not None:
            np.save(filename,val)
