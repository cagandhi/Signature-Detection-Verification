# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:27:44 2018

@author: ABD17
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image

class Rect:
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = 0

    def setArea(self, area):
        self.area = area
    def getArea(self):
        return self.area
    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
    def addPadding(self, imgSize, padding):
        self.x -= padding
        self.y -= padding
        self.w += 2 * padding
        self.h += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w > imgSize[0]:
            self.w = imgSize[0] - self.x
        if self.y + self.h > imgSize[1]:
            self.h = imgSize[1] - self.y



signature = cv2.imread('083656.jpg')
cv2.imshow('Original',signature)
signature = cv2.resize(signature, (0,0), fx=2, fy=2)

def getPageFromImage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bImg = cv2.medianBlur(src = gImg, ksize = 11)
    bImg = gImg.copy()

    threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    # There is no page in the image
    if len(contours) == 0:
        print('No Page Found')
        return img

    maxRect = Rect(0, 0, 0, 0)
    for contour in contours:
        # Detect edges
        # Reference - http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        # currentArea = cv2.contourArea(contour)

        # check if length of approx is 4
        if len(corners) == 4 and currentArea > maxRect.getArea():
            maxRect.set(x, y, w, h)
            print(cv2.isContourConvex(contour))
            # maxRect.setArea(currentArea)

    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points = contour)
        if (x > maxRect.x and x < maxRect.x + maxRect.w) and (y > maxRect.y and y < maxRect.y + maxRect.h):
                contoursInPage += 1

    maxContours = 5
    if contoursInPage <= maxContours:
        print('No Page Found')
        return img

    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]



def getSignatureFromPage(img):
    imgSize = np.shape(img)
    img = cv2.resize(img,(2000,2000))

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#    gImg = cv2.GaussianBlur(gImg, (7, 7), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
#    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
    gImg = cv2.GaussianBlur(gImg, (5, 5), 0)
    
    
#    gImg = cv2.medianBlur(src = gImg, ksize=5)
    

    
    threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # Close the image to fill blank spots so blocks of text that are close together (like the signature) are easier to detect
    # Signature usually are wider and shorter so the strcturing elements used for closing will have this ratio
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (5, 5))
    cannyImg = cv2.morphologyEx(src = cannyImg, op = cv2.MORPH_CLOSE, kernel = kernel)
#    scontimg=cv2.imshow('cannyImg',cv2.resize(cannyImg, (0,0), fx=0.5, fy=0.5))
    cv2.imshow('cannyImg',cv2.resize(cannyImg, (0,0), fx=0.4, fy=0.4))

    # findContours is a distructive function so the image pased is only a copy
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    contImg=cv2.drawContours(img, contours,-1,(0,255,0),3);
#    scontImg = cv2.resize(contImg, (0,0), fx=0.5, fy=0.5) 
    cv2.imshow('ContImg',cv2.resize(contImg, (0,0), fx=0.4, fy=0.4))
    cv2.startWindowThread()

    maxRect = Rect(0, 0, 0, 0)
    maxCorners = 0
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.01 * epsilon, True)
        x, y, w, h = cv2.boundingRect(points = contour)
        
        if len(corners) > maxCorners:
            maxCorners = len(corners)
            maxRect.set(x, y, w, h)

    # Increase the bounding box to get a better view of the signature
    maxRect.addPadding(imgSize = imgSize, padding = 10)

    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]

def getSignature(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # minBlockSize = 3
    # maxBlockSize = 101
    # minC = 3
    # maxC = 101
    #
    # bestContourNo = 1000000
    # bestBlockSize = 0
    # bestC = 0
    #
    # for c in range(minC, maxC, 2):
    #     for bs in range(minBlockSize, maxBlockSize, 2):
    #         mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = bs, C = c)
    #         rmask = cv2.bitwise_not(mask)
    #         _, contours, _ = cv2.findContours(image = rmask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours) > 15 and len(contours) < bestContourNo:
    #             bestContourNo = len(contours)
    #             bestBlockSize = bs
    #             bestC = c

    # blockSize = 21, C = 10

    # TODO throw error if blockSize is bigger than image
    blockSize = 50
    C = 10
    if blockSize > imgSize[0]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[0] - 1
        else:
            blockSize = imgSize[0]

    if blockSize > imgSize[1]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[1] - 1
        else:
            blockSize = imgSize[1]

    mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = C)
    rmask = cv2.bitwise_not(mask)

    return cv2.bitwise_and(signature, signature, mask=rmask)

#signature = getPageFromImage(img = signature)
signature = getSignatureFromPage(img = signature)
#signature = getSignature(img = signature)
cv2.startWindowThread()
cv2.imshow('Signature', signature)

cv2.waitKey();
