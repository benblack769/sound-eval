#include "file-to-wav.h"
#include <iostream>

using namespace std;

int main(){
    WavData data = print_sound_file("output.wav");
    for(int j = 0; j < data.num_frames(); j++){
        for(int i = 0;i < data.num_channels(); i++){
            cout << data.get(j,i) << "\t";
        }
        cout << "\n";
    }
}
