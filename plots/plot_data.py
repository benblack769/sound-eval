import matplotlib.pyplot as plt
import sys
import numpy as np
import bisect

def average(ls):
    return sum(ls)/len(ls)

class TimeData:
    def __init__(self,filename):
        self.filename = filename
        self.load_data()

    def load_data(self):
        with open(self.filename) as file:
            file_lines = file.readlines()

        datalist = []
        times = []
        for line in file_lines:
            vals = [float(val) for val in line.split()]
            datalist.append(np.array(vals[1:]))
            times.append(vals[0])
        self.data = np.transpose(np.vstack(datalist))
        self.times = np.array(times)

    def average_n_steps(self,n):
        '''width =  self.data.shape[0]
        leng = self.data.shape[1]
        num_after = leng % n
        if num_after != 0:
            num_avgs = 1 + leng // n
            start = np.reshape(self.data[:,:-num_after],(width,leng,num_avgs-1))
            avg_calc = np.average(start,2)
            all_avg = np.hstack(avg_calc,np.average(self.data[:,-num_after:]))
        else:
            num_avgs = leng // n
            start = np.reshape(self.data,(width,leng,num_avgs))
            all_avg = np.average(start,2)
        self.data = all_avg'''

        if len(self.times) < 2:
            return
        timedif = self.times[1] - self.times[0]

        new_data = []
        for line in self.data:
            start_i = 0
            sum_time = 0
            data_row = []
            new_times = []
            for i in range(len(self.times)):
                if sum_time > n or i == len(self.times)-1:
                    new_times.append(average(self.times[start_i:i]))
                    data_row.append(average(line[start_i:i]))

                    start_i = i
                    sum_time = 0
                sum_time += timedif
            new_data.append(np.array(data_row))
        self.data = np.vstack(new_data)
        self.times = np.array(new_times)

    def crop_window(self,start_t,end_t):
        list_t = self.times
        start = bisect.bisect_left(list_t,start_t)
        end = bisect.bisect_right(list_t,end_t)
        self.times = self.times[start:end]
        self.data = self.data[:,start:end]

    def filter_lines(self,linelist):
        linelist.sort()
        new_data = []
        for l in linelist:
            assert l >= 0 and l < self.data.shape[0], 'linelist contains indexes out of range'
            new_data.append(self.data[l])
        self.data = new_data

    def get_diff(self):
        self.data = self.data[:,1:]-self.data[:,:-1]
        self.times = self.times[1:]

    def show_plot(self):
        for d in self.data:
            plt.plot(self.times,d)
        plt.show()

    def save_plot(self,filename):
        for d in self.data:
            plt.plot(self.times,d)
        plt.savefig(filename)
        plt.close()
