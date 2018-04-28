#include <sndfile.h>
#include <cstdio>
#include <iostream>
#include <cassert>
#include "file-to-wav.h"
using namespace std;

WavData::WavData(std::vector<float> in_data,int num_frames, int num_channels){
    data = in_data;
    channels = num_channels;
    frames = num_frames;
    assert(data.size() == num_frames * num_channels);
}

WavData convert_to_data (SNDFILE * infile, int channels,int frames)
{
    vector<float> data(channels*frames);
    int readcount = sf_readf_float (infile, data.data(), frames);
    cout << "readcount: " << readcount << endl;
    cout << "frames: " << frames << endl;
    if(readcount != frames){
        cout << "bad file info" << endl;
        exit(1);
    }
    return WavData(data,frames,channels);
} /* convert_to_text */

WavData print_sound_file(std::string filename){
    SF_INFO		sfinfo ;
    SNDFILE * infile = sf_open (filename.c_str(), SFM_READ, &sfinfo);
	if (infile == NULL){
    	printf ("Not able to open input file %s.\n", filename.c_str()) ;
		puts (sf_strerror (NULL)) ;
		exit (1) ;
	} ;
	return convert_to_data (infile, sfinfo.channels, sfinfo.frames) ;
}
