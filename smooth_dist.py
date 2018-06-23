import numpy as np
import matplotlib.pyplot as plot

#import tensorflow as tf
#from pixnn import discretized_mix_logistic_loss

DATA_SIZE = 1000
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
wavelength_distribution_songs()
exit(0)
inputs = tf.placeholder(tf.float32, shape=(None, 1))

targets = tf.placeholder(tf.int32, shape=(None, NUM_BUCKETS))

with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
