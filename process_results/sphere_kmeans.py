from spherecluster import SphericalKMeans
import argparse
import pandas
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def calc_logit_regress_stats(inputs,outputs,plot_name,K_SIZE):
    skm = SphericalKMeans(n_clusters=K_SIZE)
    skm.fit(inputs)
    input_labels = skm.labels_

    out_keys = list(set(outputs))
    out_idx_mapping = {out:idx for idx,out in enumerate(out_keys)}
    #out_key_list = [out_idx_mapping[key] for key in out_keys]
    print(out_idx_mapping)

    k_center_labels = [[0]*K_SIZE for x in range(len(out_keys))]
    for k_c,lab in zip(input_labels,outputs):
        out_idx = out_idx_mapping[lab]
        k_center_labels[out_idx][k_c] += 1

    k_center_labels = np.asarray(k_center_labels)
    ind = np.arange(K_SIZE)
    plots = []
    bottom = np.zeros(K_SIZE)
    for x in range(len(out_keys)):
        plots.append(plt.bar(ind, k_center_labels[x],bottom=bottom))
        bottom += k_center_labels[x]

    plt.title('Song genres in spherical k-means clusters')
    plt.xticks(ind, ["K"+str(i+1) for i in range(K_SIZE)])
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend(plots, out_keys)
    plt.savefig(plot_name)


def run_stats(doc_csv,doc_vecs,csv_col,out_name,k_size):
    result = doc_csv[csv_col]# == "drilling"
    #result = doc_csv['genre_top']
    #uniques = set(result)
    #mapping = {item:idx for idx,item in enumerate(uniques)}
    #result = np.asarray([mapping[item] for item in result])
    calc_logit_regress_stats(doc_vecs,result,out_name,k_size)


def read_file(filename):
    with open(filename) as file:
        return file.read()

def order_csv(filename_list, data_csv):
    val_dataframe = pandas.DataFrame(data={
        "filename":filename_list
    })
    unique_data = data_csv.drop_duplicates(subset="filename")
    joined_metadata = val_dataframe.merge(unique_data,on="filename",how="left",sort=False)
    return joined_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect document statistics")
    parser.add_argument('document_csv', help='Document csv.')
    parser.add_argument('order_filename_list', help='list of filenames that are in the same order as the vectors.')
    parser.add_argument('vectors_npy', help='Vectors npy doc.')
    parser.add_argument('--centers', dest='k_size', default="8",
                help='number of centers for k-means')
    parser.add_argument('--output', dest='out_name', default="sphere_kmeans.png",
                help='path for output plot')
    parser.add_argument('--csv_collum', dest='csv_col',
                help='csv collumn to plot')

    args = parser.parse_args()

    k_size = int(args.k_size)
    data_csv = pandas.read_csv(args.document_csv)
    data_vectors = np.load(args.vectors_npy)
    filename_list = [os.path.normpath(fname)[:-4] for fname in read_file(args.order_filename_list).strip().split("\n")]
    ordered_csv = order_csv(filename_list, data_csv)
    run_stats(ordered_csv,data_vectors,args.csv_col,args.out_name,k_size)
