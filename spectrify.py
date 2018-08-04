import matplotlib.pyplot as plt
import tensorflow.contrib.signal as tfsignal
import tensorflow as tf
import numpy as np
from file_processing import mp3_to_raw_data

LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 7600.0

def plot_spectrogram(Sxx,time_segment_size):
    Sxx = Sxx.transpose()
    t = np.arange(Sxx.shape[1])*time_segment_size
    f = np.arange(Sxx.shape[0])
    plt.pcolormesh(t, f, Sxx)
    #plt.imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower')
    plt.ylabel('Frequency [Mel bins]')
    plt.xlabel('Time [seconds]')
    plt.show()

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
            pow_spec_res = sess.run([spectrogram],feed_dict={
                signals: raw_sound.reshape((1,len(raw_sound))),
            })
            spectrogram_list.append(pow_spec_res[0][0])

    return spectrogram_list

def calc_mp3_spectrogram(mp3_filename, num_mel_bins, samplerate, time_frame_len):
    raw_data = mp3_to_raw_data(mp3_filename,samplerate)
    res = None if raw_data is None else  calc_spectrogram(raw_data, num_mel_bins, samplerate, time_frame_len)
    return res

def calc_spectrogram(raw_sound, num_mel_bins, samplerate, time_frame_len):
    return spectrify_audios([raw_sound],num_mel_bins,samplerate, time_frame_len)[0]
