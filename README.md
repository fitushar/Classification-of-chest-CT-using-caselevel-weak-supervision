# Classification-of-chest-CT-using-caselevel-weak-supervision
Classification of chest CT using caselevel weak supervision

This Repository is for the work "Classification of chest CT using caselevel weak supervision" present in SPIE Medical Imaging, 2019, San Diego, California, United States.(https://doi.org/10.1117/12.2513576)

# Citation
```
Ruixiang Tang, Fakrul Islam Tushar, Songyue Han, Rui Hou, Geoffrey D.
Rubin, Joseph Y. Lo, "Classification of chest CT using case-level weak
supervision," Proc. SPIE 10950, Medical Imaging 2019: Computer-Aided
Diagnosis, 1095017 (13 March 2019); doi: 10.1117/12.2513576

@inproceedings{10.1117/12.2513576,
author = {Ruixiang Tang and Fakrul  Islam Tushar and Songyue Han and Rui Hou and Geoffrey D. Rubin and Joseph Y. Lo},
title = {{Classification of chest CT using case-level weak supervision}},
volume = {10950},
booktitle = {Medical Imaging 2019: Computer-Aided Diagnosis},
editor = {Kensaku Mori and Horst K. Hahn},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {303 -- 310},
keywords = {Machine Learning, Atelectasis, Edema, Pneumonia, Nodule, Weak Supervised Classification, Chest CT},
year = {2019},
doi = {10.1117/12.2513576},
URL = {https://doi.org/10.1117/12.2513576}
}
```
# Abstract 
Our goal is to investigate using only case-level labels extracted automatically from radiology reports to construct a multi-disease classifier for CT scans with deep learning method. We chose four lung diseases as a start: atelectasis, pulmonary edema, nodule and pneumonia. From a dataset of approximately 5,000 chest CT cases from our institution, we used a rule-based model to analyze those radiologist reports, labeling disease by text mining to identify cases with those diseases. From those results, we randomly selected the following mix of cases: 275 normal, 170 atelectasis, 175 nodule, 195 pulmonary edema, and 208 pneumonia. As a key feature of this study, each chest CT scan was represented by only 10 axial slices (taken at regular intervals through the lungs), and furthermore all slices shared the same label based on the radiology report. So the label was weak, because often disease will not appear in all slices. We used ResNet-50[1] as our classification model, with 4-fold cross-validation. Each slice was analyzed separately to yield a slice-level performance. For each case, we chose the 5 slices with highest probability and used their mean probability as the final patient-level probability. Performance was evaluated using the receiver operating characteristic (ROC) area under the curve (AUC). For the 4 diseases separately, the slice-based AUCs were 0.71 for nodule, 0.79 for atelectasis, 0.96 for edema, and 0.90 for pneumonia. The patient-based AUC were 0.74 for nodule, 0.83 for atelectasis, 0.97 for edema, and 0.91 for pneumonia. We backprojected the activations of last convolution layer and the weights from prediction layer to synthesize a heat map [2] . This heat map could be an approximate disease detector, also could tell us feature patterns which ResNet-50 focus on.

![images](https://github.com/fitushar/Classification-of-chest-CT-using-caselevel-weak-supervision/blob/master/featured2.png)
![pimages](https://github.com/fitushar/Classification-of-chest-CT-using-caselevel-weak-supervision/blob/master/featured.png)
