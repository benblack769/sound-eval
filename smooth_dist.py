import numpy as np
import matplotlib.pyplot as plot
from WeightBias import DenseLayer

import tensorflow as tf
#from pixnn import discretized_mix_logistic_loss

DATA_SIZE = 100000
TEST_SIZE = 200

NUM_BUCKETS = 50
BUCKET_START = -1
BUCKET_END = 2
BUCKET_RANGE = BUCKET_END - BUCKET_START

norm_data = np.random.normal(0.3,0.1,DATA_SIZE)
test_data = np.random.normal(0.3,0.1,TEST_SIZE)

def number_to_bucket(data):
    float_bucket = (data - BUCKET_START) / BUCKET_RANGE
    index = int(NUM_BUCKETS * float_bucket)
    return index

def bucket_to_vector(bucket_index):
    res = np.zeros(NUM_BUCKETS)
    res[bucket_index] = 1
    return res

def bucket_to_number(bucket_index):
    return (bucket_index / NUM_BUCKETS) * BUCKET_RANGE + BUCKET_START

def plot_numbers(numbers):
    plot.hist(numbers, density=1, bins=50)
    plot.xlabel('Weight')
    plot.ylabel('Probability')
    plot.show()

def plot_vector(vector):
    ind = (np.arange(0, NUM_BUCKETS))
    plot.bar(ind,vector,width=0.5)
    plot.show()

def numbers_to_vectors(numbers):
    return np.stack(bucket_to_vector(number_to_bucket(num)) for num in numbers)

def wavelength_distribution_songs():
    import process_fma_files
    all_data = process_fma_files.get_raw_data(50000)
    data_selection = np.random.choice(all_data,size=10000,replace=False)
    bucket_vals = np.sum(numbers_to_vectors(data_selection),axis=0)
    #plot_vector(bucket_vals)
    plot_numbers(data_selection)


#numbers_form = np.sum(numbers_to_vectors(norm_data),axis=0)
#plot_vector(numbers_form)
#wavelength_distribution_songs()

def get_dist_train_pred_fns(inputs,targets):
    HIDDEN_SIZE_1 = 50
    HIDDEN_SIZE_2 = 120
    HIDDEN_SIZE_3 = 80
    ADAM_learning_rate = 0.001


    in_to_hid = DenseLayer("in_to_hid",1,HIDDEN_SIZE_1)
    hid1_to_hid2 = DenseLayer("hid1_to_hid2",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    hid2_to_hid3 = DenseLayer("hid2_to_hid3",HIDDEN_SIZE_2,HIDDEN_SIZE_3)
    hid_to_out = DenseLayer("hid_to_out",HIDDEN_SIZE_3,NUM_BUCKETS)

    hid_layer_1 = tf.nn.relu(in_to_hid.calc_output(inputs))
    hid_layer_2 = tf.nn.relu(hid1_to_hid2.calc_output(hid_layer_1))
    hid_layer_3 = tf.nn.relu(hid2_to_hid3.calc_output(hid_layer_2))
    out_layer = tf.sigmoid(hid_to_out.calc_output(hid_layer_3))

    final_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = targets,
        logits = out_layer,
    )

    prediction = tf.nn.softmax(out_layer)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)

    train_op = optimizer.minimize(final_cost)
    return prediction,train_op

def randomly_pull_ammount(nparr,batch_size):
    indicies = np.random.choice(len(nparr), batch_size)
    return np.stack(nparr[idx] for idx in indicies)

def run_training(sess,train_op,inputs,targets):
    BATCH_SIZE = 32

    train_input = np.reshape(norm_data,(len(norm_data),1))
    train_expected = numbers_to_vectors(train_input)
    randomly_pull_ammount(train_input,BATCH_SIZE)

    for epoc in range(5):
        for x in range(0,len(train_input)-BATCH_SIZE,BATCH_SIZE):
            sess.run(train_op,feed_dict={
                inputs: randomly_pull_ammount(train_input,BATCH_SIZE),
                targets: randomly_pull_ammount(train_expected,BATCH_SIZE),
            })

def run_test(sess,pred_op,inputs,targets):
    test_input = np.reshape(test_data,(len(test_data),1))
    test_expected = numbers_to_vectors(test_input)
    sample = 5
    result = sess.run(pred_op,feed_dict={
        inputs: test_input[sample:sample+5],
        targets: test_expected[sample:sample+5],
    })
    print(result)




def run_all():
    inputs = tf.placeholder(tf.float32, shape=(None, 1))
    targets = tf.placeholder(tf.float32, shape=(None, NUM_BUCKETS))

    pred_op,train_op = get_dist_train_pred_fns(inputs,targets)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("initted",flush=True)
        run_training(sess,train_op,inputs,targets)
        print("training finished",flush=True)
        run_test(sess,pred_op,inputs,targets)

run_all()

exit(0)
