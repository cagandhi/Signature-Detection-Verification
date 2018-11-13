import preproc
import features
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu   # For finding the threshold for grayscale to binary conversion


def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    # print("(ann)")
    img = preproc.preproc(path, display=display)
    ratio = features.Ratio(img)
    centroid = features.Centroid(img)
    eccentricity, solidity = features.EccentricitySolidity(img)
    skewness, kurtosis = features.SkewKurtosis(img)
    # print("(svm)")
    aspect_ratio, bounding_rect_area, hull_area, contour_area=features.get_contour_features(img,display=display)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis, aspect_ratio, hull_area/bounding_rect_area, contour_area/bounding_rect_area)
    return retVal

def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1], temp[6], temp[7], temp[8])
    return features

def makeCSV():
    if not(os.path.exists('data/Features')):
        os.mkdir('data/Features')
        print('New folder "Features" created')
    if not(os.path.exists('data/Features/Training')):
        os.mkdir('data/Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('data/Features/Testing')):
        os.mkdir('data/Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = 'data/valid/'
    # forged signatures path
    fpath = 'data/invalid/'
    for person in range(1, 30):
        per = ('00'+str(person))[-3:]
        print('Saving features for person id-',per)
        
        with open('data/Features/Training/training_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,aspect_ratio,hull_area,contour_area,output\n')
            # Training set
            for i in range(0,3):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0,3):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open('data/Features/Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,aspect_ratio,hull_area,contour_area,output\n')
            # Testing set
            for i in range(3, 5):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(3,5):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')

if __name__ == "__main__":
    makeCSV()