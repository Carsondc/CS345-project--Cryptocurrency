import numpy as np

def test123():
    print('no')

def test321():
    print('yes')


def test666(listlist):
    # filename='data/qsarbiodegradation/biodeg.csv'
    # qsar = pd.read_csv(filename,sep=';', header=None)
    listlist = np.asarray(listlist, dtype=float)
    X = listlist[:,:-1]
    y = listlist[:,-1]
    return X, y