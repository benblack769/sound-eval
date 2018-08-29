import numpy as np
import tensorflow as tf
import shutil

class OutputVectors:
    def __init__(self,num_songs,vector_size):
        self.init_val = init_val = np.random.standard_normal((num_songs,vector_size)).astype('float32')/vector_size

        self.num_songs = num_songs
        self.vector_size = vector_size
        self.all_vectors = tf.Variable(init_val,name="output_vectors")

    def initialize(self,sess):
        sess.run(self.all_vectors.assign(self.init_val))

    def get_index_rows(self,indicies):
        return tf.gather(self.all_vectors,indicies,axis=0)#,shape=(indicies.shape[0],self.vector_size))

    def get_vector_values(self,sess):
        return sess.run(self.all_vectors)

    def load_vector_values(self,sess,values):
        sess.run(self.all_vectors.assign(values))

class ResultsTotal:
    def __init__(self,vectors_dir):
        self.vectors_dir = vectors_dir

    def get_filepath(self,timestep):
        return "{path}vector_at_{timestep}.npy".format(path=self.vectors_dir,timestep=timestep)

    def load_file(self,timestep):
        return np.load(self.get_filepath(timestep))

    def save_file(self,data,timestep):
        np.save(self.get_filepath(timestep),data)

    def clear_files(self):
        shutil.rmtree(self.vectors_dir)
