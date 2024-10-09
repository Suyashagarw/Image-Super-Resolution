# Image Super Resolution

## Abstract
In this project, we present an implementation of the Super-Resolution Convolutional Neural Network (SRCNN) using the PyTorch framework for Image Super Resolution. The SRCNN model architecture comprises multiple convolutional layers designed to learn the mapping between low-resolution and high-resolution image patches. The project workflow includes data loading and preprocessing, model definition, training loop, and evaluation on test data. Our dataset consists of pairs of low-resolution and high-resolution images, which are used to train and validate the SRCNN model. During training, Mean Squared Error (MSE) loss is minimized using the Adam optimizer. The model's performance is assessed using commonly used image quality metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM). Additionally, we provide visualizations of input low-resolution images, high-resolution ground truth, and model outputs to qualitatively assess the effectiveness of our approach. We demonstrate the ability of our SRCNN model to effectively enhance the visual quality of low-resolution images, showcasing its potential applications in various domains, including surveillance, medical imaging, and satellite imaging. Overall, this project contributes to the advancement of ISR techniques and provides insights into the practical implementation of deep learning models for image enhancement tasks.

## Introduction
Image Super Resolution is a fundamental task in computer vision aimed at enhancing the visual quality of low-resolution images. Applications include medical imaging, aiding early detection and diagnosis of diseases, and surveillance and security. Both applications often suffer from inadequate quality and detail, diminishing their effectiveness for detailed analysis and recognition tasks. While improving hardware can be considered, there are cost and physical limitations, making software-based enhancements such as super-resolution more practical and appealing.

Traditional approaches to image enhancement often rely heavily on manual processes and a high level of expertise. By leveraging deep learning through Convolutional Neural Networks, it is possible to automate image enhancement. This project revolves around creating an image resolution enhancer, leveraging deep learning to automate the process of image resolution enhancement. Our main objective is to enhance image clarity and detail for improved image recognition and analysis.

## Problem Specification
This project's objective is to develop a solution that enhances the resolution of blurry, low-resolution images into clearer, high-resolution images leveraging deep learning techniques.

### Requirements
- The enhancement process must not distort or change the original image content.
- The solution must be implemented using the PyTorch framework.
- The solution must not use any pre-trained models.
- The solution should process images in a reasonable time frame.

### Constraints
- The project is constrained by the computational resources available for training and testing the deep learning model.
- The effectiveness of the model will depend on both the model architecture and the quality of the dataset used.

## Design Details

### Model Architecture
For this project, the Super Resolution Convolutional Neural Network (SRCNN) architecture will be explored. The SRCNN model consists of three layers: a feature extraction layer, a non-linear mapping layer, and a reconstruction error layer.

![SRCNN Architecture](link-to-your-image)

The feature extraction layer extracts features from the low-resolution input image. This layer employs convolutional filters for feature extraction. The extracted features are then passed through a non-linear mapping layer to map the low-resolution feature into high-resolution features. The non-linearity of the layer captures the complex transformations for accurate super resolution, typically achieved using a ReLU activation function applied with another convolutional layer. 

The final step of the SRCNN is to reconstruct the high-resolution image via the reconstruction layer, which aggregates the feature maps into a single high-resolution image.

### Dataset and Preprocessing
The dataset used was the Image Super Resolution Dataset, containing a wide range of images to ensure model robustness. The dataset includes both low-resolution and their high-resolution counterparts. The low-resolution images serve as inputs, while the high-resolution images are used as targets. Images were loaded as tensors and normalized. The dataset was further separated into training, validation, and testing data.

- Train Set: 70%
- Validation Set: 15%
- Test Set: 15%

### Hyperparameter Tuning
Hyperparameters were tuned based on past research and some experimentation.

1. **Transformations**:
   - Resize Dimension: ((256, 256))
   - Mean values for normalization: [0.485, 0.456, 0.406]
   - Std values for normalization: [0.229, 0.224, 0.225]

2. **SRCNN Model Architecture**:
   - Kernel sizes and paddings for each convolutional layer:
     - First: kernel size: 9, padding: 4
     - Second: kernel size: 1, padding: 0
     - Third: kernel size: 5, padding: 2
   - ReLU activation function for the first and second layers.

3. **Training Parameters**:
   - Learning Rate: 0.001
   - Epochs: 20
   - Batch Size: 32
   - Shuffle: True
   - Optimizer: Adam
   - Loss Function: MSE

### Model Framework and Training
The neural network was developed with Python using PyTorch. The model weights were initialized to random small values. The training dataset was used to calculate the forward pass of the neural network, and the output was compared with the target higher-resolution image to determine the loss. Gradients were calculated through back propagation, and weights were updated using the Adam optimizer after each batch. The training was carried out for 20 epochs.

## Numerical Experiments and Results

### Model Parameters
- Learning Rate: 0.001
- Epochs: 20
- Batch Size: 32
- Optimizer: Adam
- Loss Function: MSE

#### Evaluation Metrics
The model effectively enhances the resolution of blurry input images, as shown by the performance metrics for different test images.

| Metric | Image Set 1 | Image Set 2 |
|--------|-------------|-------------|
| MSE    | 0.042       | 0.064       |
| PSNR   | 13.752      | 11.921      |
| SSIM   | 0.852       | 0.767       |

The average performance metrics from evaluating the entire test dataset are as follows:

| Metric          | Average Value |
|-----------------|---------------|
| Average MSE     | 0.028         |
| Average PSNR    | 16.600        |
| Average SSIM    | 0.818         |

## Conclusion 
Through this project, we have learned the importance of data preprocessing, model architecture design, training, and evaluation for achieving high-quality super-resolved images. We observed how factors such as hyperparameters, loss functions, and optimization algorithms affect the performance of the SRCNN model. The project demonstrated the significance of quantitative metrics such as PSNR and SSIM, as well as qualitative assessment, in evaluating model performance.

## References
1. Dong, Chao, Loy, Chen Change, He, Kaiming, Tang, Xiaoou. (2014). Image Super-Resolution Using Deep Convolutional Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38. [DOI: 10.1109/TPAMI.2015.2439281](https://doi.org/10.1109/TPAMI.2015.2439281).
