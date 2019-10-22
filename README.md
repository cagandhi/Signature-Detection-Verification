# Signature-detection-verification
A conceptual model to detect and verify signatures on bank cheques. This was my project in the "Computer Vision" course.

We devised an OCR based approach to localize signatures along with Connected Components, Linesweep algorithm & geometric features with Support Vector Machine (SVM), Artificial Neural Networks (ANN) classification to verify the authenticity of signatures with a 91% hit rate.

Signature_Detection folder contains codes for different approaches:<br> Contour features-based, OCR + Connected Components labelling, OCR + LineSweeping algorithm.<br><br>
Signature_verification folder contains codes for SVM and ANN classification.

Usage:
1. Navigate to folder whose approach you want to run.<br>
  For OCR approach - run 'acheque.py'<br>
  For Connected Components approach - run 'cclabel.py'<br>
  For LineSweep algorithm, run 'lineSweepDetect.py'<br>
  For Contour feature based algorithm, run 'signature_detection.py'.

2. For execution of signature verification approaches,<br>
  For SVM based classification, run 'run.py' <br>
  For ANN based classification, run 'test.py'
