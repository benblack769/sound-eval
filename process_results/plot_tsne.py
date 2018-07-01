import numpy as np
import sklearn.manifold

def plot_tsne(data):
    tsne = sklearn.manifold.TSNE()
    #print(data.shape)
    #exit(1)
    transformed_data = tsne.fit_transform(data)

    return transformed_data
    #print(data)


if __name__ == "__main__":
    vectors_folder = "../relu_repo_results/"
    vectors_file = "vector_at_1851.npy"
    plot_tsne_results = "process_results/relu_repo_results_1851_2d.npy"
    data = np.load(vectors_folder+vectors_file)
    data2d = plot_tsne(data)
    np.save(plot_tsne_results,data2d)
