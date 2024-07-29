<div align="center">
  <p>
    <a href="https://github.com/ultralytics/assets/releases/tag/v8.2.0" target="_blank">
      <img width="100%" src="https://production-media.paperswithcode.com/datasets/59a87a08-a2ae-4f94-8cd5-a8c4774f97f4.png" alt="YOLO Vision banner"></a>
  </p>
<br>
</div>

# NeoPolyp

## Overview

**BKAI-IGH NeoPolyp-Small** is a publicly available dataset released by the BKAI Research Center, Hanoi University of Science and Technology, in collaboration with the Institute of Gastroenterology and Hepatology (IGH), Vietnam. This dataset is specifically designed for the tasks of polyp segmentation and classification.

## Dataset Description

### NeoPolyp-Small

- **Total Images**: 1200 images
  - **WLI Images**: 1000
  - **FICE Images**: 200
- **Training Set**: 1000 images
- **Test Set**: 200 images
- **Classification**: Polyps are classified into neoplastic (red) and non-neoplastic (green) classes.

## Acknowledgments

This dataset is collected thanks to the project VINIF.2020.DA17 funded by Vingroup Innovation Foundation. We extend our gratitude to IGH for collecting and annotating the data.

## Evaluation Metric

The primary evaluation metric for this competition is the mean Dice coefficient and cross-entropy loss, which measures the pixel-wise agreement between the predicted segmentation and the ground truth.

**Dice Coefficient Formula**
The Dice Coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

\text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}

where \( X \) is the predicted set of pixels and \( Y \) is the ground truth.

**Cross Entropy Loss Formula**

The Cross Entropy Loss is used to measure the performance of a classification model whose output is a probability value between 0 and 1. The formula is given by:

\[ \text{Cross Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p(y_i)) + (1 - y_i) \log(1 - p(y_i)) \right] \]

where \( y_i \) is the true label and \( p(y_i) \) is the predicted probability.

## Training phase
![image](https://github.com/user-attachments/assets/f3b3d2e1-473b-422c-abe4-eb88d68fdedc)
![image](https://github.com/user-attachments/assets/ef0e4d50-d936-4083-9fd8-587f59a2f9e6)

## Testing phase
![image](https://github.com/user-attachments/assets/8b18e49d-1f20-42d7-91c5-e47aed54d108)
