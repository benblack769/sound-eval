import theano
import numpy as np
import shutil
import time
import os

SHARED_DIR = "saved_weights/"

def _save_share(path,sharevar):
    fname = os.path.join(path,sharevar.name)
    assert not os.path.exists(fname), "file already exists: '{}'. Are your shared variable names unique?".format(fname)
    val = sharevar.get_value()
    np.save(fname,val)

def _load_share(path,share_val):
    try:
        fname = os.path.join(path,share_val.name)+".npy"
        share_val.set_value(np.load(fname))
    except IOError:
        return

def save_shared(path,share_list):
    os.makedirs(path)
    for sh in share_list:
        _save_share(path,sh)

def delete_shared(path):
    if os.path.exists(path):
        shutil.rmtree(path)

class RememberSharedVals:
    def __init__(self,holder_name,update_freq=10.0):
        self.name = holder_name
        self.vals = []
        self.update_freq = update_freq
        self.time_last_saved = time.process_time()
        self.has_saved = False
    def names(self):
        return [val.name for val in self.vals]
    def add_shared_val(self,share_val):
        assert share_val.name not in self.names(), "cannot add two shared values with the same name to a single RememberSharedVals object"
        _load_share(self.path(),share_val)
        self.vals.append(share_val)
    def add_shared_vals(self,sh_val_list):
        for val in sh_val_list:
            self.add_shared_val(val)
    def should_update(self):
        time_since_save = time.process_time() - self.time_last_saved
        return self.update_freq < time_since_save
    def vals_updated(self):
        if self.should_update():
            self.force_update()
    def path(self):
        return os.path.join(SHARED_DIR,self.name)
    def share_save_fn(self,fn):
        def newfn(*args):
            out = fn(*args)
            self.vals_updated()
            return out
        return newfn
    def force_update(self):
        name = SHARED_DIR+self.name
        tempname = SHARED_DIR+"_tmp_"+self.name

        #assert not os.path.exists(name) or self.has_saved, "weights are already saved! delete {} folder to continue".format(name)

        os.makedirs(SHARED_DIR, exist_ok=True)
        save_shared(tempname,self.vals)
        delete_shared(name)
        os.rename(tempname,name)
        self.time_last_saved = time.process_time()
        self.has_saved = True
