# Towards Visually Explaining Statistical Tests

## Overview
This project aims to demystify the "black-box" nature of statistical tests by visualizing the differences between two statistically distinct groups at both the sample and feature levels. Given two groups that are statistically different, our goal is to identify which specific samples and features contribute to these differences.

To determine if two groups are statistically different, we utilize a non-parametric two-sample statistical test introduced by [Matthias Kirchler,2020]. In this approach, the two groups of data are embedded in the last hidden layer of a neural network, and a multivariate location test is conducted to assess whether both groups map to the same location in this representation space.

To identify the samples driving the differences between the two groups, we propose an *influence score* that quantifies the contribution of each sample to the observed differences. Both analytical and empirical results confirm that samples with the highest influence scores are the primary contributors to the differences between the groups.

To visualize the features that differentiate the two groups, we employ a gradient-based method. By backpropagating the statistical test signal to the input, we highlight the specific pixels responsible for the differences.

This method can be applied to various contexts, such as identifying the samples and features that distinguish groups of medical images—e.g., one group with a genetic mutation and another without.

## Project structure
- **simclr.py**: script to train a self-supervised learning based on simcler to obtain embeddings.
- **models.py**: script including the models used in simclr model.
- **data.py**: script that defines groups of Diabetic Rethinopathy (here we defined two groups healthy and unhealthy group).
- **aptos_data.py**: script that prepares Aptos data for pretarining using simclr.
- **embeddingtest.py**: script which can be used to see if two groups are statistically different or not.
- **sampleimpr.py**: assings an influence score to each sample.
- **samexclude.py**: exclude samples with highest influence score and measure the p-value.
- **gradcam.py**: script that uses Grad-Cam to backprobagate test statitic.
- **vis2samtestdr.py**: scripts that is used to visualise the pixels that make difference between two groups.
- **overlay.py**: is used to overlay heatmpas obtained from vis2samtestdr.py on images.
- **utils.py**: containes utility functions for visualisation.

## Data Access
We pretrain a SimCLR model on the **train split** of the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset, which can be downloaded from Kaggle.
Each image is preprocessed by cropping to the circular region and resizing to **448×448** pixels.

For **visualizing important features and representative samples**, we use the pretrained SimCLR model to extract embeddings from the **test set** of the [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) dataset, also available on Kaggle.

## How to run
1- **Clone the repository**:
   ```bash
   git clone https://github.com/Mjavan/ExplainingStatisticalTest.git
    
   cd ExplainingStatisticalTests
   ```
    
2- **Install dependencies**:
  `pip install -r requirements.txt`

3- **To get the embeddings**:
   `python src/simclr.py`

4- **To get the influence scores**:
   `python src/sampleimpr.py`

5- **To visualaise features that make differences between two groups**:
   `python src/vis2samtestdr.py`

6- **Additional scripts**:
   - Generate and overlay heatmaps using `python src/overlay.py`
   - Check if two groups are statistically different or not using `python src/embeddingtest.py`
## Dependencies
- Pytorch
- Numpy
- Matplotlib
- Scikit-learn
- Seaborn

## License
This project is licensed under the MIT License.


  
   
 

