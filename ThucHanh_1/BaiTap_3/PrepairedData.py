from sklearn.datasets import fetch_lfw_people
from skimage.feature import local_binary_pattern
import numpy as np

def save():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = np.array([]).reshape(0,1850)
    for image in lfw_people.images:
        lbt_image = local_binary_pattern(image,P=8,R=0.5).flatten()
        X = np.append(X,[lbt_image],axis=0)
    np.save(file='data.npy',arr=X)
#save()
def load():
    return  np.load('data.npy')