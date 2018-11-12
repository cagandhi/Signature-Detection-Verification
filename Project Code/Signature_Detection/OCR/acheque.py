try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import csv 
import cv2
import os
import numpy as np

# If you don't have tesseract executable in your PATH, include the following:
# (?# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>')
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
# print(pytesseract.image_to_string(Image.open('Cheque083654.jpg')))

# # French text image to string
# # print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# # Get bounding box estimates
files = [f for f in os.listdir('.') if os.path.isfile(f)]
please=0;
sign=0;
above=0;
for filename in files:
    # filename = 'Cheque 309105.jpg'
    if ".jpg"in filename:
        print ("file ", filename)
        img = cv2.imread(filename)
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # img = cv2.filter2D(img, -1, kernel)
        # img = cv2.resize(img,(0,0),fx=2,fy=2)
    else:
        continue;
    h, w, _ = img.shape # assumes color image

    # Image.open('Cheque083654.jpg'))
    # print(pytesseract.image_to_boxes(img)


    # run tesseract, returning the bounding boxes
    # boxes = pytesseract.image_to_boxes(img) # also include any config options you use

    # # draw the bounding boxes on the image
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # show annotated image and wait for keypress

    # Get verbose data including boxes, confidences, line and page numbers
    data=pytesseract.image_to_data(Image.open(filename))
    # print (data)
    pleaseCd=[0, 0, 0,0];
    aboveCd=[0, 0, 0, 0];

    for d in data.splitlines():
        d = d.split('\t')
        if len(d)==12:
            
            if d[11].lower()=='please':
                pleaseCd[0]=int(d[6]);
                pleaseCd[1]=int(d[7]);
                pleaseCd[2]=int(d[8]);
                pleaseCd[3]=int(d[9]);
                please=please+1;
            if d[11].lower()=='sign':
                sign=sign+1;
            if d[11].lower()=='above':
                aboveCd[0]=int(d[6]);
                aboveCd[1]=int(d[7]);
                aboveCd[2]=int(d[8]);
                aboveCd[3]=int(d[9]);
                above=above+1;

    lengthSign=aboveCd[0]+aboveCd[3]-pleaseCd[0]
    scaleY=2;
    scaleXL=2.5;
    scaleXR=0.5;
    
    # print ("here "+str(lengthSign))
    lengthSignCd=[0, 0, 0, 0];

    # print ("here ",pleaseCd)
    lengthSignCd[0]=int(pleaseCd[0]-lengthSign*2.5);
    lengthSignCd[1]=int(pleaseCd[1]-lengthSign*2);

    # print ("here ",lengthSignCd)
    img = cv2.rectangle(img, (lengthSignCd[0],lengthSignCd[1] ), (lengthSignCd[0]+int((scaleXL+scaleXR+1)*lengthSign), lengthSignCd[1]+int(scaleY*lengthSign)), (255, 255, 255), 2)
    cropImg=img[lengthSignCd[1]:lengthSignCd[1]+int(scaleY*lengthSign),lengthSignCd[0]:lengthSignCd[0]+int((scaleXL+scaleXR+1)*lengthSign)]
    
    path = '/Users/miteshgandhi/Desktop/CVPROJ/Project Code/OCR/Results'

    s1 = 'Cropped_' + filename
    cv2.imwrite(os.path.join(path, s1), cropImg)

# print(please ," ", sign," ",above)