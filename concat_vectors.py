import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenates two .npy vectors along their 2nd axis")
    parser.add_argument('vec1', help='Path to first vector to concatenate.')
    parser.add_argument('vec2', help='Path to second vector to concatenate.')
    parser.add_argument('out_vec', help='Path to output filename where .npy result will be stored.')

    args = parser.parse_args()

    vec1 = np.load(args.vec1)
    vec2 = np.load(args.vec2)
    concat = np.concatenate([vec1,vec2],axis=1)
    np.save(args.out_vec,concat)
