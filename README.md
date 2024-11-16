
# Scalable Rice Grain Classification Using CNN and Apache Spark

## Project Overview
This project focuses on developing a scalable system for classifying rice grains into distinct varieties using Convolutional Neural Networks (CNNs) integrated with Apache Spark. The classification process leverages the Kaggle Rice Image Dataset, which contains 75,000 high-quality images across five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. By utilizing Apache Spark’s distributed computing capabilities, this project addresses the challenges of processing and training on large datasets efficiently while delivering a robust and accurate classification model.

---

## Objectives
- Automate rice classification using CNNs for high accuracy and efficiency.
- Leverage Apache Spark for distributed data processing and model training.
- Implement data augmentation, hyperparameter tuning, and regularization techniques to enhance performance.
- Enable scalability to process large datasets across multiple machines.
- Utilize advanced techniques like Grad-CAM for model interpretability.

---

## Dataset
- **Name**: Rice Image Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data)  
- **Description**: The dataset includes 75,000 images, evenly distributed across five rice varieties. Each image is high-resolution and labeled for supervised learning tasks.  
- **Preprocessing**: Images are resized to 64x64 pixels and normalized for input into the CNN model.

---

## Features
1. **CNN Architecture**:
   - Three convolutional layers with increasing filters (32, 64, 128).
   - Max-pooling layers to reduce dimensionality.
   - Fully connected dense layers for classification with dropout regularization.
   - Softmax activation for multi-class output.

2. **Distributed Processing**:
   - Apache Spark for parallel data loading and preprocessing.
   - TensorFlowOnSpark for distributed model training across multiple nodes.

3. **Techniques for Improved Generalization**:
   - Data augmentation with rotations, flips, zooms, and brightness adjustments.
   - Regularization using dropout layers and L2 weight penalties.
   - Early stopping to prevent overfitting.

4. **Performance Metrics**:
   - Accuracy, precision, recall, F1-score, and confusion matrix.
   - Cross-validation to ensure robustness.




