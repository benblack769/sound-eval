import argparse
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
import sklearn.linear_model
from sklearn import svm    			# To fit the svm classifier\

PROPORTION_TEST = 0.2

def get_test_train(data1,data2):
    assert 0 < PROPORTION_TEST < 1
    choose_size = int((1-PROPORTION_TEST)*len(data1))
    train_keys = np.random.choice(np.arange(0,len(data1)),size=choose_size,replace=False)
    train_key_set = set(train_keys)
    test_keys = np.asarray([x for x in range(0,len(data1)) if x not in train_key_set])

    train_data1 = data1[train_keys]
    train_data2 = data2[train_keys]
    test_data1 = data1[test_keys]
    test_data2 = data2[test_keys]
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
        self.IN_LEN = len(inputs[0])
        self.OUT_LEN = max(expected)+1
        self.model = Sequential([
            Dense(self.OUT_LEN, input_shape=(self.IN_LEN,)),
            #Activation('relu'),
            #Dense(self.OUT_LEN),
            Activation('softmax'),
        ])
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.fit(inputs,one_shot_vec(expected,self.OUT_LEN),epochs=100,batch_size=32)

    def predict(self, inputs):
        return self.model.predict_classes(inputs)

    def score(self, inputs, expected):
        score = self.model.evaluate(inputs, one_shot_vec(expected,self.OUT_LEN), batch_size=32)
        return score

def calc_logit_regress_stats(inputs,outputs):
    (train_inputs,train_outputs),(test_inputs,test_outputs) = get_test_train(inputs,outputs)
    logit_model = SoftmaxFitter()
    #logit_model = svm.SVC(kernel='linear')
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

def run_stats(doc_csv,doc_vecs):
    new_doc_vecs = np.concatenate([np.maximum(doc_vecs,0),np.maximum(-doc_vecs,0)],axis=1)
    #result = doc_csv['classID']# == "drilling"
    result = doc_csv['genre_top']
    uniques = set(result)
    mapping = {item:idx for idx,item in enumerate(uniques)}
    result = np.asarray([mapping[item] for item in result])
    calc_logit_regress_stats(doc_vecs,result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect document statistics")
    parser.add_argument('document_csv', help='Document csv.')
    parser.add_argument('vectors_npy', help='Vectors npy doc.')

    args = parser.parse_args()

    data_csv = pandas.read_csv(args.document_csv)
    data_vectors = np.load(args.vectors_npy)
    run_stats(data_csv,data_vectors)
