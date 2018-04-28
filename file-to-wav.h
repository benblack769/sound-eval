#include <string>
#include <vector>

class WavData{
public:
    WavData(std::vector<float> in_data,int num_frames, int num_channels);
    float get(int fr,int ch){
        return data[fr*channels+ch];
    }
    int num_channels(){
        return channels;
    }
    int num_frames(){
        return frames;
    }
protected:
    std::vector<float> data;
    int channels;
    int frames;
};

WavData print_sound_file(std::string filename);
