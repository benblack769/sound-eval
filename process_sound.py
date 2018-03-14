import soundfile as sf
import matplotlib.pyplot as plt
sig, samplerate = sf.read('output.wav')

print(sig.shape)
print(samplerate)
print(sig[5000:5010])
