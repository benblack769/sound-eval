import numpy as np
import argparse
import yaml
import os
import process_many_files
import tensorflow as tf
from linearlizer import Linearlizer
from helperclasses import OutputVectors
from calc_all_vecs import run_spec_list
import random

gpu_config = tf.ConfigProto(
    device_count = {'GPU': int(False)}
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

def get_train_batch(word_vecs,cmp_vecs,all_cmp_vecs,local_var_vecs, LOCAL_WINDOW_SIZE, WINDOW_SIZE, wordvecs_len):
    NUM_WORDS_PER_BATCH = 5
    local_vec_size = wordvecs_len - LOCAL_WINDOW_SIZE - WINDOW_SIZE
    half_batch_size = local_vec_size * NUM_WORDS_PER_BATCH
    #use_local_vecs = local_var_vecs[:local_vec_size]
    local_ids = tf.tile(tf.range(local_vec_size,dtype=tf.int32), (NUM_WORDS_PER_BATCH*2,))
    #print(local_ids.shape)
    word_ids = local_ids + tf.random_uniform((half_batch_size*2,),dtype=tf.int32,minval=0,maxval=LOCAL_WINDOW_SIZE)
    true_cmp_ids = word_ids[:half_batch_size] + tf.random_uniform((half_batch_size,),dtype=tf.int32,minval=1,maxval=WINDOW_SIZE)
    false_cmp_ids = tf.random_uniform((half_batch_size,),dtype=tf.int32,minval=0,maxval=all_cmp_vecs.shape[0])

    true_cmps = tf.gather(cmp_vecs,true_cmp_ids,axis=0)
    false_cmps = tf.gather(all_cmp_vecs,false_cmp_ids,axis=0)
    all_cmps = tf.concat([true_cmps,false_cmps],axis=0)

    orign_vecs = tf.gather(word_vecs,word_ids,axis=0)

    local_vecs = tf.gather(local_var_vecs,local_ids,axis=0)

    valid_is_trues = tf.zeros((half_batch_size,),dtype=tf.float32)
    invalid_is_trues = tf.ones((half_batch_size,),dtype=tf.float32)
    is_correct = tf.concat([valid_is_trues,invalid_is_trues],axis=0)

    return orign_vecs,all_cmps,local_vecs,is_correct

def calc_all_locals(spec_list,config,model_path):

    linearlizer = Linearlizer(config['NUM_MEL_BINS'], config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'])

    all_cmp_vecs_np = calc_all_compare_vecs(spec_list,config,model_path,linearlizer)
    all_compare_vecs = np.concatenate(all_cmp_vecs_np)
    all_word_vecs = run_spec_list(spec_list,config,model_path,linearlizer)

    all_cmp_vecs_var = tf.constant(all_compare_vecs,dtype=tf.float32)

    word_vecs = tf.placeholder(tf.float32, [None, config['OUTPUT_VECTOR_SIZE']])
    word_vecs_size = tf.placeholder(tf.int32, [])
    cur_cmp_vecs = tf.placeholder(tf.float32, [None, config['OUTPUT_VECTOR_SIZE']])

    linearlizer = Linearlizer(config['NUM_MEL_BINS'], config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'])

    max_spec_len = max(len(spec) for spec in spec_list)

    local_vec_var = OutputVectors(max_spec_len, config['OUTPUT_VECTOR_SIZE'])

    origin_compare, cross_compare, local_vecs, is_same_compare = get_train_batch(word_vecs,cur_cmp_vecs,all_cmp_vecs_var,local_vec_var.all_vectors,config['LOCAL_VEC_WINDOW_SIZE'],config['WINDOW_SIZE'],word_vecs_size)

    loss = linearlizer.loss_vec_computed(origin_compare, cross_compare, local_vecs, is_same_compare)

    SGD_learning_rate = 10.0
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=config['ADAM_learning_rate'])
    optim = optimizer.minimize(loss)

    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        local_vecs_list = []
        for idx in range(len(spec_list)):
            local_vec_var.initialize(sess)
            for x in range(100):
                opt_val, loss_val = sess.run([optim, loss],feed_dict={
                    word_vecs:all_word_vecs[idx],
                    cur_cmp_vecs:all_cmp_vecs_np[idx],
                    word_vecs_size:len(all_word_vecs[idx])
                })
            print(loss_val)
            print("new vec calculated")
            local_var_len = len(all_word_vecs[idx]) - config['LOCAL_VEC_WINDOW_SIZE'] - config['WINDOW_SIZE']
            loc_vecs = local_vec_var.get_vector_values(sess)[:local_var_len]
            print(loc_vecs)
            local_vecs_list.append(loc_vecs)

    return local_vecs_list

def make_dirs(paths):
    for path in paths:
        pathdir = os.path.dirname(path)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

def calc_all_vectors(source_dir, dest_dir, model_path, config):
    all_filenames = process_many_files.get_all_paths(source_dir,"npy")
    random.shuffle(all_filenames)
    all_filenames = all_filenames[:10]
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
