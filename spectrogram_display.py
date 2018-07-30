import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from file_processing import mp3_to_raw_data
from spectrogram_compute import stft,plotstft

SAMPLERATE = 16000
fs = 10e3
def sine_wave():
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500*np.cos(2*np.pi*0.25*time)
    carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise
    return x

def load_audio():
    return mp3_to_raw_data("../fma_small/000/000211.mp3",SAMPLERATE)

#print(spectrify(load_audio()))
f, t, Sxx  = signal.spectrogram(load_audio(),fs=SAMPLERATE,nperseg=2**6)
#print(spectrify())
print(Sxx.shape)
#print(load_audio().shape)
#print(Sxx.transpose()[100])
#print(f,t)
#exit(1)
#res = plotstft(load_audio(),SAMPLERATE,2**6)
#res = stft(load_audio(),2**6)
#timebins, freqbins = np.shape(spec)
#print(res)
print(t.shape)
print(f.shape)
print(t)
print(f)
plt.pcolormesh(t, f, np.log(Sxx+0.00001))
#plt.imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
