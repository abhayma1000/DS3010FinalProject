# Import Libraries
import tensorflow
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import transformers
import pickle

# NLP Packages
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup

# Iterate through each category number type
batch_cats = [15, 30]

for batch_cat in batch_cats:
    # Iterate through each packet type
    batch_packets = ['Packet 1', 'Packet 2', 'Packet 3']

    for batch_packet in batch_packets:
        # Load train & validate datasets
        train_dataset = torch.load('/home/natomasetti/DS3010/Yelp_Tensor_Datasets/'+ str(batch_cat) + '_' + batch_packet + '/Train_Tensor_Dataset.pt', weights_only = False)
        val_dataset = torch.load('/home/natomasetti/DS3010/Yelp_Tensor_Datasets/'+ str(batch_cat) + '_' + batch_packet + '/Validate_Tensor_Dataset.pt', weights_only = False)
        
        # Batch size for dataloader
        batch_size = 32

        # DataLoaders for our training and validation sets.
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )

        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

        # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = batch_cat, # The number of output labels 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            problem_type = "multi_label_classification" # Defaults loss function to BCEWithLogitsLoss
        )

        # Tell model to run on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(),
                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

        # Define loss function compatible with multi-label classification
        criterion = nn.BCEWithLogitsLoss()

        # Number of training epochs (4 recommended by BERT authors)
        epochs = 4

        # Total number of training steps (number of batches x number of epochs)
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        ###########################
        # Train with training set #
        ###########################
        def train(model, iterator, optimizer, criterion, device, scheduler, epoch):
            
            # Enter train mode
            model.train()
            train_loss = 0

            # Number of iterations equal to total train dataset / batch size
            for step, batch in enumerate(iterator):
                #Print progress in epoch
                print(f"Progress: {step+1}/{len(iterator)}", end='\r')
                
                # Parse iterator tensor dataset for important information
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                
                # Generate prediction
                optimizer.zero_grad()
                
                # Compute gradients and update weights
                loss = criterion(output.logits, b_labels) # BCEWithLogitsLoss has sigmoid
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Accumulate train loss
                train_loss += loss
            
            # Print completed result
            print()
            print('Train Loss: %f' % (train_loss))
            return train_loss

        #############################
        # Validate with testing set #
        #############################
        def test(model, iterator, optimizer, criterion, device, epoch):

            # Enter evaluation mode
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for step, batch in enumerate(iterator):
                
                    # Print progress in epoch
                    print(f"Progress: {step+1}/{len(iterator)}", end='\r')
                    
                    # Parse iterator tensor dataset for important information
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)
                    
                    # Generate prediction
                    output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)
                    prob = output.logits.sigmoid()   # BCEWithLogitsLoss has sigmoid
                    
                    # Record processed data count
                    total += (b_labels.size(0)*b_labels.size(1))

                    # Take the index of the highest prob as prediction output
                    THRESHOLD = 0.7
                    prediction = prob.detach().clone()
                    prediction[prediction > THRESHOLD] = 1
                    prediction[prediction <= THRESHOLD] = 0
                    correct += prediction.eq(b_labels).sum().item()

                print()
            
            # Print completed result
            acc = 100.*correct/total
            print('Correct: %i  / Total: %i / Test Accuracy: %f' % (correct, total, acc))
            return acc

        # Iterate through each epoch and call train and validation functions
        for e in range(epochs):
            
            print(f"\n===== Epoch {e+1}/{epochs} =====")
            start_time = time.time()
            # training
            print("Training started ...")
            train(model, train_dataloader, optimizer, criterion, device, scheduler, e)

            # validation testing
            print("Testing started ...")
            test(model, validation_dataloader, optimizer, criterion, device, e)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {e+1} Elapsed Time: {elapsed_time:.4f} seconds")
        
        # Save each model to be loaded later
        model.save_pretrained('/home/natomasetti/DS3010/Models/BERT_' + str(batch_cat) + '_' + batch_packet)