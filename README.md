# Signature-detection-verification
A conceptual model to detect and verify signatures on bank cheques. This was my project in the "CSP 502: Computer Vision" course at Ahmedabad University.

We devised an OCR based approach to localize signatures along with Connected Components, Linesweep algorithm & geometric features with Support Vector Machine (SVM), Artificial Neural Networks (ANN) classification to verify the authenticity of signatures with a 91% hit rate.

Reference for connected components code: [https://github.com/spwhitt/cclabel](https://github.com/spwhitt/cclabel)

### Description of files:
Signature_Detection folder contains codes for different approaches: Contour features-based, OCR + Connected Components labelling, OCR + LineSweeping algorithm.<br><br>
Signature_verification folder contains codes for SVM and ANN classification.

# Usage
1. Install dependencies via: `pip install -r requirements.txt`
2. For execution of signature detection approaches<br>
* OCR approach - run `python acheque.py`<br>
* Connected Components approach - run `python cclabel.py`<br>
* LineSweep algorithm, run `python lineSweepDetect.py`<br>
* Contour feature based algorithm, run `python signature_detection.py`
3. For execution of signature verification approaches,<br>
* For SVM based classification, run `python run.py` <br>
* For ANN based classification, run `python test.py`
