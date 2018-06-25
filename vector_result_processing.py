import os
import numpy as np
import shutil
import tensorflow as tf

class ResultsTotal:
    def __init__(self,vectors_dir):
        self.vectors_dir = vectors_dir
        
    def get_filepath(self,timestep):
        return "{path}vector_at_{timestep}.npy".format(path=self.vectors_dir,timestep=timestep)

    def save_file(self,data,timestep):
        np.save(self.get_filepath(timestep),data)

    def clear_files(self):
        shutil.rmtree(self.vectors_dir)
