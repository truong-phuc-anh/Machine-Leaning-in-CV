import glob
from skimage.io import imread
import numpy as np
from skimage.transform import resize
from skimage.feature import hog

def save():
    X = np.array([]).reshape(0, 756)
    for filename in glob.glob('cars_train/*.jpg'):
        im = imread(filename,as_grey=True)
        im = resize(im,(50,75), mode='constant')
        x = hog(im, block_norm='L2', orientations=3)
        X = np.append(X,[x],axis=0)
    np.save(file='data.npy', arr=X)
#save()

def load():
    return np.load('data.npy')