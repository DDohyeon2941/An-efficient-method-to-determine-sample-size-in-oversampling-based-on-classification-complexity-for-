# Efficient Oversampling Based on Classification Complexity

This repository contains the implementation of an efficient oversampling method designed to handle imbalanced datasets. By leveraging classification complexity measures, this method generates synthetic samples only where they are most needed, reducing oversampling size, improving computational efficiency, and minimizing the risk of overfitting.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview
Imbalanced datasets are common in many real-world applications and pose significant challenges for machine learning models. Conventional oversampling methods, such as SMOTE, often create excessive minority class samples, which can lead to overfitting. Our method optimizes the number of synthetic samples generated by considering the **classification complexity** of the dataset, ensuring that fewer, but more impactful, samples are generated.

### Key Features:
- **Reduced Oversampling Size**: Unlike traditional methods that generate a balanced number of samples, this approach creates fewer synthetic samples, reducing overfitting and computational costs.
- **Complexity-Aware Sampling**: Incorporates both feature-based and neighborhood-based classification complexity measures to guide the generation of new samples.
- **Boosting Integration**: Works with ensemble techniques like **SMOTEBoost**, **RAMOBoost**, **WOTBoost**, and **K-means SMOTEBoost**, making it suitable for boosting algorithms that are sensitive to noisy or excessive samples.

## Data
The following imbalanced datasets were used to validate this method, sorted by imbalance ratio (IR):

1. **Ionosphere**: 350 samples, 34 attributes, IR = 1.80
2. **KC1**: 1,212 samples, 21 attributes, IR = 2.85
3. **Ecoli**: 336 samples, 7 attributes, IR = 8.60
4. **Optical Digits**: 5,620 samples, 64 attributes, IR = 9.14
5. **Satimage**: 6,435 samples, 36 attributes, IR = 9.28
6. **Abalone**: 4,177 samples, 7 attributes, IR = 9.68
7. **Spectrometer**: 531 samples, 93 attributes, IR = 10.80
8. **Isolet**: 7,797 samples, 617 attributes, IR = 12.00
9. **US Crime**: 1,994 samples, 100 attributes, IR = 12.29
10. **Oil**: 937 samples, 27 attributes, IR = 21.85
11. **Wine Quality**: 3,961 samples, 11 attributes, IR = 21.90
12. **Glass**: 213 samples, 9 attributes, IR = 22.67
13. **Yeast Me2**: 1,453 samples, 8 attributes, IR = 27.49
14. **Letter Img**: 18,668 samples, 16 attributes, IR = 28.17
15. **Mammography**: 7,849 samples, 6 attributes, IR = 29.90
16. **PC1**: 4,901 samples, 6 attributes, IR = 42.76


## Methodology

### Classification Complexity Measures

#### 1. **Feature-Based Complexity Measures**

Feature-based measures assess the discriminative power of individual features in distinguishing between classes.

### **F1 (Maximum Fisher's Discriminant Ratio)**

**Original Definition**:  
F1 measures the ability of features to separate two classes based on their means and variances. A **higher F1** value in this context indicates **greater difficulty in separating** the two classes, meaning the classes have more overlap and are harder to distinguish, suggesting worse performance.

**Modification**:  
The F1 measure now specifically focuses on features where the separability between the minority and majority classes is lower (i.e., higher complexity). This helps ensure that attention is given to regions of the feature space where the minority class struggles the most, guiding synthetic sample generation to where it’s most needed.
**F2 (Class Overlap Measure)**:

  **Original Definition**:  
  F2 assesses the degree of overlap between the classes across each feature. It identifies how much the feature values of the two classes mix together, indicating the complexity of separating them.

  **Modification**:  
  The F2 measure is modified to include **boundary samples**, capturing the complexity at the decision boundaries. This is especially important in imbalanced datasets, where minority class samples often reside near the majority class. Including these boundary samples improves the ability to identify areas where more synthetic samples are needed.

#### 2. **Neighborhood-Based Complexity Measures**

Neighborhood-based measures focus on the local distribution of samples around each instance, particularly the distances to neighbors within and across classes.

- **N1 (Intra/Extra Class Distance Ratio)**:

  **Original Definition**:  
  N1 evaluates how close samples from the same class are to each other compared to samples from different classes. A higher N1 value indicates that the samples are well-separated within their own class, reducing classification complexity.

  **Modification**:  
  The N1 measure now considers **k-nearest neighbors** instead of just the nearest neighbor, providing a more detailed view of local class separation. Additionally, this measure focuses specifically on the minority class samples, ensuring the method captures the complexity of separating minority samples from the majority class.

- **N2 (1-Nearest Neighbor Error Rate)**:

  **Original Definition**:  
  N2 calculates how often a sample is misclassified by its nearest neighbor, providing insight into the local complexity of class boundaries. A higher N2 value means more frequent misclassifications, indicating greater complexity.

  **Modification**:  
  N2 is adapted to use **k-nearest neighbors** instead of just one nearest neighbor. This modification offers a more comprehensive understanding of local complexity, especially for minority class samples, helping to target synthetic sampling efforts in regions where minority samples are frequently misclassified.

### Summary of Modifications:

- **F1** now focuses on features with lower separability for the minority class.
- **F2** includes boundary samples to better capture class overlap in critical decision boundary regions.
- **N1** is modified to use k-nearest neighbors and focuses specifically on minority class samples to reflect the difficulty of separating minority instances from the majority.
- **N2** is extended to use k-nearest neighbors and focuses on the error rate for minority class samples, offering a more nuanced view of complexity in imbalanced datasets.


## Results
Our experiments across 16 different imbalanced datasets show that the proposed oversampling method achieves superior or comparable performance to traditional oversampling techniques while significantly reducing the number of synthetic samples. This not only reduces computational time but also mitigates the risk of overfitting caused by excessive sampling.

### Key Benefits:
- **Efficiency**: Reduces computational costs by generating fewer synthetic samples.
- **Generalization**: Improves model performance by focusing synthetic sample generation on areas where classification is most difficult, avoiding overfitting.
- **Robustness**: Particularly effective when combined with ensemble learning methods, such as boosting algorithms, where excessive minority samples can lead to overfitting.

## Contributing

We welcome contributions to this project. Please submit pull requests or open issues for any bugs or enhancements.

## References
For more details, please refer to the full paper on [ELSEVIER](https://doi.org/10.1016/j.eswa.2021.115442).
