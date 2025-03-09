#Import Libraries
import matplotlib.pyplot as plt
import sklearn
import tensorflow
import numpy as np
import pandas as pd
import time
import torch
import pickle
import transformers

#NLP Packages
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score

# Iterate through each number of categories
batch_cats = ['15', '30']

for batch_cat in batch_cats:
        # Iterate through each packet type
        batch_packets = ['Packet 1', 'Packet 2', 'Packet 3']
        for batch_packet in batch_packets:
                # Load model and test tensor dataset
                model = BertForSequenceClassification.from_pretrained("/home/natomasetti/DS3010/Models/BERT_" + batch_cat + "_" + batch_packet)
                test_tensor_dataset = torch.load('/home/natomasetti/DS3010/Yelp_Tensor_Datasets/' + batch_cat + "_" + batch_packet +'/Test_Tensor_Dataset.pt', weights_only = False)
                
                # Load MultiLabelBinarizer for test dataset
                with open('/home/natomasetti/DS3010/Yelp_Tensor_Datasets/'+ batch_cat + '_' + batch_packet + '/test_mlb.pkl', 'rb') as f:
                        mlb = pickle.load(f)
                
                # Initialize batch_size
                batch_size = 32

                # Set test DataLoader
                test_dataloader = DataLoader(
                        test_tensor_dataset, # The validation samples.
                        sampler = SequentialSampler(test_tensor_dataset), # Pull out batches sequentially.
                        batch_size = batch_size # Evaluate with this batch size.
                        )
                
                # Use GPU if available
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                
                # Store predictions and true labels
                predictions = []
                true_labels = []

                # Iterate through each batch
                for batch in test_dataloader:
                        # Parse iterator tensor dataset for important information
                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)
                        with torch.no_grad():
                                #Generate prediction
                                output = model(b_input_ids, 
                                                token_type_ids=None, 
                                                attention_mask=b_input_mask)
                                logits = output.logits
                                logits = logits.detach().cpu().numpy()
                                prob = output.logits.sigmoid()   # BCEWithLogitsLoss has sigmoid
                                
                                # Set threshold to determine predictions based on probability (0.5 preformed the best)
                                THRESHOLD = 0.5
                                prediction = prob.detach().cpu().numpy()
                                prediction[prediction > THRESHOLD] = 1
                                prediction[prediction <= THRESHOLD] = 0
                                
                                # Save true labels, extend prediction and true labels list
                                true_label = b_labels.detach().cpu().numpy()
                                predictions.extend(list(prediction))
                                true_labels.extend(list(true_label))

                # Convert lists to array
                predictions = np.array(predictions)
                true_labels = np.array(true_labels)
                
                # Save total length
                total_len = len(predictions)
                
                # Can be used to see text values for each true labels and predicted labels
                predictions_it = mlb.inverse_transform(predictions)
                true_labels_it = mlb.inverse_transform(true_labels)
                
                # Micro-averaged scores
                micro_precision = precision_score(true_labels, predictions, average = 'micro')
                micro_recall = recall_score(true_labels, predictions, average = 'micro')
                micro_F1 = f1_score(true_labels, predictions, average = 'micro')

                # Micro-averaged scores
                weight_precision = precision_score(true_labels, predictions, average = 'weighted')
                weight_recall = recall_score(true_labels, predictions, average = 'weighted')
                weight_F1 = f1_score(true_labels, predictions, average = 'weighted')

                # Print evaluation metrics
                print(f'\nCategories: {batch_cat}, Packets: {batch_packet}, Total Businesses: {str(total_len)}')

                print(f'Micro-averaged precision: {micro_precision}, Weighted-averaged precision: {weight_precision}\nMicro-averaged recall: {micro_recall}, Weighted-averaged recall: {weight_recall}\nMicro-averaged F1: {micro_F1}, Weighted-averaged F1: {weight_F1}')