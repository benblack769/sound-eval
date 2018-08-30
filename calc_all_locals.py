import numpy as np
import argparse
import yaml
import os
import sys
import process_many_files
import tensorflow as tf
from linearlizer import Linearlizer
from helperclasses import OutputVectors
import random

gpu_config = tf.ConfigProto(
    device_count = {'GPU': int(True)},
)

def calc_all_compare_vecs(spec_list,config,model_path,linearlizer):
    vecs = tf.placeholder(tf.float32, [None, config['NUM_MEL_BINS']])

    comp_vec = linearlizer.compare_vector(vecs)

    with tf.Session() as sess:
        linearlizer.load(sess, model_path)
        compvecs_list = []
        for spec in spec_list:
            val = sess.run(comp_vec,feed_dict={
                vecs:spec
            })
            compvecs_list.append(val)

    return compvecs_list

def get_train_batch(word_vecs,cmp_vecs, config):
    NUM_WORDS_PER_BATCH = 5

    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    STEPS_BETWEEN_WINDOWS = config['STEPS_BETWEEN_WINDOWS']
    INVALID_BEFORE_RANGE_START = config['INVALID_BEFORE_RANGE_START']
    WINDOW_GAP_RANGE = config['WINDOW_GAP_RANGE']
    INVALID_BEFORE_RANGE_VAR = config['INVALID_BEFORE_RANGE_VAR']
    #BATCH_SIZE = config['SAMPLES_PER_SONG']

    word_vecs = word_vecs#[0]
    word_vec_len = tf.shape(word_vecs)[0]

    START_LOC = INVALID_BEFORE_RANGE_START + INVALID_BEFORE_RANGE_VAR + STEPS_IN_WINDOW
    END_LOC = word_vec_len - (STEPS_IN_WINDOW*2 + STEPS_BETWEEN_WINDOWS + WINDOW_GAP_RANGE)

    local_vec_size = END_LOC - START_LOC
    half_batch_size = local_vec_size * NUM_WORDS_PER_BATCH
    #use_local_vecs = local_var_vecs[:local_vec_size]

    origin_starts = tf.tile(tf.range(START_LOC,END_LOC,dtype=tf.int32), (NUM_WORDS_PER_BATCH*2,))#tf.random_uniform((BATCH_SIZE,1),dtype=tf.int32,minval=START_LOC,maxval=END_LOC)
    valid_compare_starts = origin_starts[:half_batch_size] + (STEPS_IN_WINDOW + STEPS_BETWEEN_WINDOWS) + tf.random_uniform((half_batch_size,),dtype=tf.int32,minval=-WINDOW_GAP_RANGE,maxval=WINDOW_GAP_RANGE+1)
    invalid_compare_start = origin_starts[half_batch_size:] + (- STEPS_IN_WINDOW - INVALID_BEFORE_RANGE_START) - tf.random_uniform((half_batch_size,),dtype=tf.int32,minval=0,maxval=INVALID_BEFORE_RANGE_VAR)
    all_compare_starts = tf.concat([valid_compare_starts,invalid_compare_start],axis=0)

    origins = tf.gather(word_vecs, origin_starts,axis=0)
    compares = tf.gather(cmp_vecs, all_compare_starts,axis=0)

    valid_is_trues = tf.zeros((half_batch_size,),dtype=tf.float32)
    invalid_is_trues = tf.ones((half_batch_size,),dtype=tf.float32)
    is_correct = tf.concat([valid_is_trues,invalid_is_trues],axis=0)

    return origins,compares,origin_starts,is_correct

def calc_all_locals(spec_list,config,model_path):
    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    STEPS_BETWEEN_WINDOWS = config['STEPS_BETWEEN_WINDOWS']
    INVALID_BEFORE_RANGE_START = config['INVALID_BEFORE_RANGE_START']
    WINDOW_GAP_RANGE = config['WINDOW_GAP_RANGE']
    INVALID_BEFORE_RANGE_VAR = config['INVALID_BEFORE_RANGE_VAR']

    SAMPLE_SIZE = config['SAMPLES_PER_SONG']
    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    VEC_SIZE = config['NUM_MEL_BINS']
    SONGS_PER_BATCH =  config['SONGS_PER_BATCH']

    input_shape = (-1,STEPS_IN_WINDOW,VEC_SIZE)

    linearlizer = Linearlizer(config['NUM_MEL_BINS'], STEPS_IN_WINDOW, config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'], input_shape, config['MAG_REGULARIZATION_PARAM'])

    spec_vecs = tf.placeholder(tf.float32, [None, config['NUM_MEL_BINS']])
    word_vecs_size = tf.shape(spec_vecs)[0]

    batches = [spec_vecs[x:word_vecs_size+x-STEPS_IN_WINDOW] for x in range(STEPS_IN_WINDOW)]

    inputs = tf.stack(batches,axis=2)

    word_vec_tfval = linearlizer.word_vector(inputs)
    cmp_vec_tfval = linearlizer.compare_vector(inputs)

    word_vecs = tf.placeholder(tf.float32, [None, config['OUTPUT_VECTOR_SIZE']])
    cmp_vecs = tf.placeholder(tf.float32, [None, config['OUTPUT_VECTOR_SIZE']])

    max_spec_len = max(len(spec) for spec in spec_list)

    local_vec_var = OutputVectors(max_spec_len, config['OUTPUT_VECTOR_SIZE'])

    origin_compare, cross_compare, local_ids, is_same_compare = get_train_batch(word_vecs,cmp_vecs,config)

    local_vecs = local_vec_var.get_index_rows(local_ids)
    print(origin_compare.shape)
    print(cross_compare.shape)
    print(local_vecs.shape)
    loss = linearlizer.loss_vec_computed(origin_compare, cross_compare, local_vecs, is_same_compare)

    SGD_learning_rate = 5.0
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=config['ADAM_learning_rate']*0.01)
    optim = optimizer.minimize(loss)

    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        linearlizer.load(sess,model_path)
        local_vecs_list = []
        for idx in range(len(spec_list)):
            local_vec_var.initialize(sess)
            word_vec_val, cmp_vec_val = sess.run([word_vec_tfval,cmp_vec_tfval],feed_dict={
                spec_vecs:spec_list[idx]
            })

            for x in range(100):
                opt_val, loss_val = sess.run([optim, loss],feed_dict={
                    word_vecs:word_vec_val,#all_word_vecs[idx],
                    cmp_vecs:cmp_vec_val
                })
                if x % 20 == 0:
                    print(loss_val)

            print("new vec calculated",idx)
            print(loss_val)
            START_LOC = INVALID_BEFORE_RANGE_START + INVALID_BEFORE_RANGE_VAR + STEPS_IN_WINDOW
            END_LOC = len(word_vec_val) - (STEPS_IN_WINDOW*2 + STEPS_BETWEEN_WINDOWS + WINDOW_GAP_RANGE)

            loc_vecs = local_vec_var.get_vector_values(sess)[START_LOC:END_LOC]
            #print(loc_vecs)
            local_vecs_list.append(loc_vecs)
            sys.stdout.flush()

    return local_vecs_list

def make_dirs(paths):
    for path in paths:
        pathdir = os.path.dirname(path)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

def calc_all_vectors(source_dir, dest_dir, model_path, config):
    all_filenames = process_many_files.get_all_paths(source_dir,"npy")
    #random.shuffle(all_filenames)
    #all_filenames = all_filenames[:20]
    source_abs_filenames = [os.path.join(source_dir,filename) for filename in all_filenames]
    dest_abs_filenames = [os.path.join(dest_dir,filename) for filename in all_filenames]

    make_dirs(dest_abs_filenames)
    source_datas = [np.load(path) for path in source_abs_filenames]
    dest_datas = calc_all_locals(source_datas,config,model_path)
    for dpath,ddata in zip(dest_abs_filenames,dest_datas):
        np.save(dpath,ddata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into a folder of .npy vector files")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('model_path', help='Path to learning model folder.')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')

    args = parser.parse_args()

    _config = yaml.safe_load(open(args.model_path+"config.yaml"))

    calc_all_vectors(args.vector_dataset,args.output_folder,args.model_path,_config)
