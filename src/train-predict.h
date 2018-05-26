#include "file-to-wav.h"

void save_vec(ostream & os, vector<float> & vec){

}
using OneDArray = vector<float>;
class TwoDArray{
    int x_size;
    int y_size;
    vector<float> data;
    TwoDArray(int in_x_size, int in_y_size):
        x_size(in_x_size),
        y_size(in_y_size),
        data(in_x_size*in_y_size){
    }
    float & get(int x, int y){
        return data[y*x_size+x];
    }
    float & operator[](int x, int y){
        return get(x,y);
    }
};

class OrdinaryLayer{
    int in_size;
    int out_size;
    vector<float> biases;
    TwoDArray weights;
public:
    OrdinaryLayer(int in_in_size, int in_out_size):
        in_size(in_in_size),
        out_size(in_out_size),
        biases(out_size,0),
        weights(in_in_size,in_out_size){
            for(int i = 0; i < )
        }
};

class TrainModel{

};

void train_model(WavData data,TrainModel & model);
