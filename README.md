# Yelp Dataset Analysis DS3010
[Nick Tomasetti](https://github.com/Ntom725), [Sam Nguyen](https://github.com/samnguyen3115), and [Abhay Mathur](https://github.com/abhayma1000)

Reach out if you have any questions

## Overview

This is our analysis of the [Kaggle Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=Dataset_User_Agreement.pdf). We set out to solve two problems:
1. Use review text to classify a business's categories. Ex: Seafood, Nail Salon
2. Predict if a business will close using review data and a business's metatdata

We utilized WPI's ARC Turing Cluster to as well as the full dataset to accomplish this.

Read our report: [Report](https://docs.google.com/document/d/1O5L8d1IwdBoGPEtg_2VV0atJ23E4Snq30WCH6bYLp5Q/edit?tab=t.0)

Look at our slides: [Slides](https://docs.google.com/presentation/d/1Dxa4xhz_3aRu029cikx_IFnciJ8sf6zP1BHDPvDv0nY/edit#slide=id.g2e3ccceb597_0_46)


## Code

Files for part 1:
* ```task1_bert_preprocessing.py```
    * Preprocess the raw yelp review data into inputs and expected value tensors
* ```task1_bert_training.py```
    * Fine-tune BERT model using the data
* ```task1_bert_analysis.py```
    * Evaluate the fine-tuned Bert model and report metrics

Files for part 2:
* ```part2preprocessing.ipynb```
    * Preprocess the raw yelp review and business data into inputs and expected value tensors
* ```part2model2.ipynb```
    * Train the DNN, Bert hybrid model and evaluate it
    * Predict which factors led to business closure predictions
    * Evaluate model

Note: This repository is not suited for simply running the code, specific setup is required


## Technologies
Used technologies:
* Torch
* BERT/transformers
* [SHAP](https://shap.readthedocs.io/en/latest/index.html)
* Numpy
* Pandas
* NLTK
* spaCy




