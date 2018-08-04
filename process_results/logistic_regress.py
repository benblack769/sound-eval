import argparse
import pandas
import numpy as np
import sklearn.linear_model

PROPORTION_TEST = 0.5

def get_test_train(data1,data2):
    assert 0 < PROPORTION_TEST < 1
    train_keys = list(np.random.choice(np.arange(0,len(data1)),size=int((1-PROPORTION_TEST)*len(data1))))
    train_key_set = set(train_keys)
    test_keys = [x for x in range(0,len(data1)) if x not in train_key_set]

    train_data1 = [data1[key] for key in train_keys]
    train_data2 = [data2[key] for key in train_keys]
    test_data1 = [data1[key] for key in test_keys]
    test_data2 = [data2[key] for key in test_keys]
    return (train_data1,train_data2),(test_data1,test_data2)

def calc_logit_regress_stats(inputs,outputs):
    (train_inputs,train_outputs),(test_inputs,test_outputs) = get_test_train(inputs,outputs)
    logit_model = sklearn.linear_model.LogisticRegression()
    logit_model.fit(train_inputs,train_outputs)
    prediction = logit_model.predict(test_inputs)

    score = logit_model.score(test_inputs,test_outputs)
    perc_correct = sum(np.asarray(test_outputs) ^ prediction)/float(len(test_outputs))
    test_composition = sum(test_outputs)/float(len(test_outputs))
    train_composition = sum(train_outputs)/float(len(train_outputs))
    print("Precentage correctly guessed: {}".format(1-perc_correct))
    print("Actual composition: {}".format(1-test_composition))
    print("Train dataset composition: {}".format(1-train_composition))
    #print("Logit score: {}".format(score))

def run_stats(doc_csv,doc_vecs):
    result = doc_csv.genre == "national_anthems"
    calc_logit_regress_stats(doc_vecs,result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect document statistics")
    parser.add_argument('document_csv', help='Document csv.')
    parser.add_argument('vectors_npy', help='Vectors npy doc.')

    args = parser.parse_args()

    data_csv = pandas.read_csv(args.document_csv)
    data_vectors = np.load(args.vectors_npy)
    run_stats(data_csv,data_vectors)
