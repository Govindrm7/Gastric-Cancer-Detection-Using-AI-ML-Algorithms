# Gastric Cancer Detection with Ensemble Learning on Digital Pathology: Use Case of Gastric Cancer on GasHisSDB Dataset

## Link for Research Paper: https://www.mdpi.com/2075-4418/14/16/1746

## Abstract
Gastric cancer is one of the deadliest cancers worldwide, and early detection through histopathological imaging can significantly improve patient outcomes. This project builds a robust deep learning ensemble model to assist in the detection of gastric cancer. By combining the strengths of multiple convolutional neural networks (CNNs) including **ResNet50**, **VGGNet16**, and **ResNet34**, high classification accuracy is achieved. The models were trained on the publicly available **Gastric Histopathology Sub-size Image Database (GasHisSDB)**, which contains images at various resolutions, ranging from 80x80 to 160x160 pixels.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#COnclusion)
- [Contributing](#contributing)

## Overview
The primary goal of this research is to develop an ensemble model that integrates multiple CNN architectures to detect gastric cancer from histopathological images. This ensemble model, trained using transfer learning, demonstrates improved accuracy over individual models, making it a powerful tool for early diagnosis.

## Dataset
The **Gastric Histopathology Sub-size Image Database (GasHisSDB)** was used for training and validation. This dataset consists of three sub-datasets:
1. **80x80-pixel images** 
2. **120x120-pixel images**
3. **160x160-pixel images**

Each sub-dataset contains a mix of normal and abnormal histopathological images.

### Preprocessing
- **Empty Patch Removal**: Empty patches with RGB intensity values above a certain threshold were removed to reduce noise.
- **Data Augmentation**: Standard data augmentation techniques were applied to increase dataset diversity.

## Methodology
The ensemble approach uses **transfer learning** with CNN architectures to extract features from images, followed by a fully connected layer for classification. The methodology is divided into the following steps:
1. **Data Preprocessing**: Removal of empty patches and performing data augmentation.
2. **Model Architecture**: The ensemble model combines **ResNet50**, **VGGNet16**, and **ResNet34**.
3. **Training and Optimization**: The models were trained using **cross-entropy loss** and optimized with **Stochastic Gradient Descent (SGD)**.
4. **Evaluation**: The ensemble model was evaluated using accuracy, sensitivity, specificity, and AUC metrics.

## Models Used
The following pre-trained models were used as base models for the ensemble:
- **Ensemble**
- **VitNet**
- **EfficientNet**
- **ResNet50**
- **VGGNet16**
- **ResNet34**

The final model is a combination of these base models.

## Training and Evaluation
- **Cross-Validation**: Stratified K-Fold cross-validation with 5 splits was used to ensure the robustness of the model.
- **Performance Metrics**:
  - **Accuracy**
  - **Jaccard Index**
  - **Recall**
  - **AUC (Area Under the Curve)**
  - **Sensitivity**
  - **Specificity**

## Results
The ensemble model achieved superior performance across all image resolutions:
- **80x80 pixels**: Accuracy of **99%**
- **120x120 pixels**: Accuracy of **99%**
- **160x160 pixels**: Accuracy of **99%**

The ensemble model significantly outperformed individual models.

## Conclusion
This research demonstrated that ensemble learning using pre-trained deep learning models like ResNet50, VGGNet16, and ResNet34 can effectively improve the accuracy of gastric cancer detection. By leveraging high-performance GPUs provided by Oracle’s Discovery Cluster, the model training process was expedited, allowing for the efficient processing of large medical image datasets. The results showed consistent improvement in accuracy, specificity, and sensitivity, making this approach viable for real-world medical applications.

##Authors
-**Govind Govind Rajesh Mudavadkar**
-**Mo Deng**
-**Salah Mohammed Awad Al-Heejawi**
-**Isha Hemant Arora**
-**Anne Breggia**
-**Bilal Ahmad**
-**Robert Christman**
-**Stephen T. Ryan**
-**Saeed Amal**
