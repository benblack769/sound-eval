import argparse
import pandas
import numpy as np
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense, Activation, Lambda
import sklearn.linear_model
from sklearn import svm    			# To fit the svm classifier\
import os
import subprocess

PROPORTION_TEST = 0.2

def get_indexes(data,keys):
    return data[keys]

def get_test_train(data1,data2):
    assert 0 < PROPORTION_TEST < 1
    choose_size = int((1-PROPORTION_TEST)*len(data1))
    train_keys = np.random.choice(np.arange(0,len(data1)),size=choose_size,replace=False)
    train_key_set = set(train_keys)
    test_keys = np.asarray([x for x in range(0,len(data1)) if x not in train_key_set])

    train_data1 = get_indexes(data1,train_keys)
    train_data2 = get_indexes(data2,train_keys)
    test_data1 = get_indexes(data1,test_keys)
    test_data2 = get_indexes(data2,test_keys)
    return (train_data1,train_data2),(test_data1,test_data2)

def one_shot(category, max_class_val):
    res = np.zeros(max_class_val,dtype=np.float32)
    res[category] = 1
    return res

def one_shot_vec(classes,max_class_val):
    return np.stack(one_shot(cat,max_class_val) for cat in classes)


class SoftmaxFitter:
    def __init__(self,batch_size=32):
        self.model = None
        self.batch_size = batch_size

    def fit(self, inputs, expected):
        self.HIDDEN_SIZE = 64
        self.BATCH_SIZE = inputs[0].shape[0]
        self.IN_SIZE = inputs[0].shape[1]
        self.OUT_LEN = max(expected)+1
        self.model = Sequential([
            Dense(self.HIDDEN_SIZE, input_shape=(self.BATCH_SIZE,self.IN_SIZE)),
            Activation('relu'),
            Dense(self.OUT_LEN),
            Lambda(lambda x: K.mean(x, axis=1), output_shape=(self.OUT_LEN,)),
            Activation('softmax'),
        ])
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.fit(inputs,one_shot_vec(expected,self.OUT_LEN),epochs=100,batch_size=128)#,verbose=0)

    def predict(self, inputs):
        return self.model.predict_classes(inputs)

    def score(self, inputs, expected):
        score = self.model.evaluate(inputs, one_shot_vec(expected,self.OUT_LEN), batch_size=32)
        return score

def pad_len(data,target_len):
    return np.concatenate([data,np.zeros((target_len-len(data),data.shape[1]))])

def load_npy_into_array(npy_paths):
    raw_data = [np.load(path) for path in npy_paths]
    longest_data = max(len(d) for d in raw_data)
    padded_data = np.stack(pad_len(d,longest_data) for d in raw_data)
    return padded_data

def calc_logit_regress_stats(npy_paths,outputs):
    npy_data = load_npy_into_array(npy_paths)
    (train_inputs,train_outputs),(test_inputs,test_outputs) = get_test_train(npy_data,outputs)
    logit_model = SoftmaxFitter()
    logit_model.fit(train_inputs,train_outputs)
    prediction = logit_model.predict(test_inputs)

    score = logit_model.score(test_inputs,test_outputs)
    print(score)
    #perc_correct = sum(np.asarray(test_outputs) ^ prediction)/float(len(test_outputs))
    #test_composition = sum(test_outputs)/float(len(test_outputs))
    #train_composition = sum(train_outputs)/float(len(train_outputs))
    #print("Precentage correctly guessed: {}".format(1-perc_correct))
    #print("Actual composition: {}".format(1-test_composition))
    #print("Train dataset composition: {}".format(1-train_composition))
    #print("Logit score: {}".format(score))

def run_stats(doc_csv,npy_paths):
    result = doc_csv['target']# == "drilling"
    #result = doc_csv['genre_top']
    uniques = set(result)
    mapping = {item:idx for idx,item in enumerate(uniques)}
    result = np.asarray([mapping[item] for item in result])
    calc_logit_regress_stats(npy_paths,result)


def read_file(filename):
    with open(filename) as file:
        return file.read()

def get_all_paths(base_folder, extension):
    find_call = ["find", ".", "-name", "*."+extension, "-type", "f"]
    file_list = subprocess.check_output(find_call,cwd=base_folder).decode('utf-8').strip().split()
    return file_list

def order_csv(filename_list, data_csv):
    val_dataframe = pandas.DataFrame(data={
        "filename":filename_list
    })
    unique_data = data_csv.drop_duplicates(subset="filename")
    joined_metadata = val_dataframe.merge(unique_data,on="filename",how="left",sort=False)
    return joined_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect document statistics")
    parser.add_argument('document_csv', help='Document csv.')
    parser.add_argument('order_filename_list', help='list of filenames that are in the same order as the vectors.')
    parser.add_argument('vector_folder', help='folders full of vector lists.')

    args = parser.parse_args()

    data_csv = pandas.read_csv(args.document_csv)
    raw_filenames = read_file(args.order_filename_list).strip().split("\n")
    csv_filenames = [os.path.normpath(fname)[:-4] for fname in raw_filenames]
    file_paths = [os.path.join(args.vector_folder,fname) for fname in raw_filenames]
    ordered_csv = order_csv(csv_filenames, data_csv)
    run_stats(ordered_csv,file_paths)
