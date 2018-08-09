import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_spectrogram(Sxx,out_file,time_segment_size):
    Sxx = Sxx.transpose()
    t = np.arange(Sxx.shape[1])*time_segment_size
    f = np.arange(Sxx.shape[0])
    plt.pcolormesh(t, f, Sxx)
    #plt.imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower')
    plt.ylabel('Elements')
    plt.xlabel('Time [seconds]')
    plt.savefig(out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 2d .npy file")
    parser.add_argument('npy_file',
                        help='Path to npy file to plot')
    parser.add_argument('output_fname',
                        help='Path to output file where plot is stored.')
    parser.add_argument('--fps', dest='frames_per_sec', default="1",
                        help='scale factor for time axis, e.g. how many vectors per second')
    args = parser.parse_args()

    vec2d = np.load(args.npy_file)
    assert len(vec2d.shape) == 2
    plot_spectrogram(vec2d,args.output_fname,float(args.frames_per_sec))
