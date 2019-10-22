# Signature-detection-verification
A conceptual model to detect and verify signatures on bank cheques. This was my project in the "Computer Vision" course.

We devised an OCR based approach to localize signatures along with Connected Components, Linesweep algorithm & geometric features with Support Vector Machine (SVM), Artificial Neural Networks (ANN) classification to verify the authenticity of signatures with a 91% hit rate.

### Description of files:
Signature_Detection folder contains codes for different approaches: Contour features-based, OCR + Connected Components labelling, OCR + LineSweeping algorithm.<br><br>
Signature_verification folder contains codes for SVM and ANN classification.

# Usage
For execution of signature detection approaches<br>
* OCR approach - run 'acheque.py'<br>
* Connected Components approach - run 'cclabel.py'<br>
* LineSweep algorithm, run 'lineSweepDetect.py'<br>
* Contour feature based algorithm, run 'signature_detection.py'.

For execution of signature verification approaches,<br>
* For SVM based classification, run 'run.py' <br>
* For ANN based classification, run 'test.py'
