
import numpy as np
from pfbayes.common.cmd_args import cmd_args
import pickle


def create_test_data(filename):

    matrix_aa = np.random.randn(cmd_args.gauss_dim, cmd_args.gauss_dim).astype(np.float32)
    matrix_a = np.random.randn(cmd_args.gauss_dim, cmd_args.gauss_dim).astype(np.float32)
    matrix_b = np.random.randn(cmd_args.gauss_dim, cmd_args.gauss_dim).astype(np.float32)
    a = np.random.randn(cmd_args.gauss_dim).astype(np.float32)
    b = np.random.randn(cmd_args.gauss_dim).astype(np.float32)

    # save
    try:
        with open(filename, 'rb') as f:
            _ = pickle.load(f)
    except Exception:
        pass
    else:
        print('ATTENTION: there exists saved test data, but now it is overwritten.')

    with open(filename, 'wb') as f:
        pickle.dump((matrix_aa, matrix_a, matrix_b, a, b), f)


if __name__ == '__main__':

    file = './test_function.pkl'
    create_test_data(file)
