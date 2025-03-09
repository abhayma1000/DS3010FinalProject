# IMPORT LIBRARIES #
import os
import json
import csv
import pandas as pd
import numpy as np
import time
import datetime
import random
import re
import nltk
import pickle
import torch

from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertConfig,BertTokenizer

### LOAD DATA ###

# LOAD FUNCTION #
def load_data(path, filename, truncate=None):
    # Open data file
    data_file = open(os.path.join(path, filename))
    data = []
    # Save trunctate number of lines
    if truncate is not None:
        for i, line in enumerate(data_file):
            data.append(json.loads(line))
            if i == truncate:
                break
    else:
        for line in data_file:
            data.append(json.loads(line))
    # Save & return DataFrame
    ret_df = pd.DataFrame(data)
    data_file.close()
    return ret_df

# LOAD BUSINESS & REVIEW DATASETS, REPORT COUNTS #
path = "/home/natomasetti/DS3010/Yelp_Data_Raw"

start_time = time.time()
reviews_df = load_data(path, "yelp_academic_dataset_review.json")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Review Load Time: {elapsed_time:.4f} seconds")

print("\nDataset Load Times:\n")
start_time = time.time()
businesses_df = load_data(path, "yelp_academic_dataset_business.json")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Business Load Time: {elapsed_time:.4f} seconds")

num_rev_load = len(reviews_df)
num_bus_load = len(businesses_df)


### CLEAN DATA ###

## REVIEWS DATA ##

# DROP NULL REVIEWS
print(f"\n\nREVIEWS DATASET:\n\nTotal Initial Reviews: {num_rev_load}")
reviews_df.dropna(subset = ['business_id', 'text'], inplace = True)
reviews_removed = num_rev_load - len(reviews_df)
print(f"Drop Null: -{reviews_removed} ({(reviews_removed/num_rev_load*100):.1f} %)")
prev_rev_length = len(reviews_df)

# FILTER OUT BUSINESSES WITH TOO FEW REVIEWS #
minimum_business_reviews = 30
reviews_df = reviews_df[reviews_df['business_id'].map(reviews_df['business_id'].value_counts()).gt(minimum_business_reviews)]
reviews_removed = prev_rev_length - len(reviews_df)
print(f"Drop Businesses Under {minimum_business_reviews} Reviews: -{reviews_removed} ({(reviews_removed/num_rev_load*100):.1f} %)")
prev_rev_length = len(reviews_df)
print(f"Total Reviews Kept: {prev_rev_length}  ({(prev_rev_length/num_rev_load*100):.1f} %)")

# REMOVE PUNCTUATION AND STOPWORDS FROM REVIEWS #
nltk.download('stopwords')
sw = stopwords.words('english')
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.,!Â¿]+", " ", text) # Replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    text = re.sub(r"http\S+", "",text) # Remove URLs
    html = re.compile(r'<.*?>')
    text = html.sub(r'',text) # Remove html tags
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^,' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') # Remove punctuations
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text) # Remove stopwords
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # Emoticons
                        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) # Removing emojis
    return text

reviews_df['text'] = reviews_df['text'].apply(lambda x: clean_text(x))
print("\nReview Text Cleaned")

# REMOVE ALL NON-NOUNS FROM REVIEWS #
nltk.download('universal_tagset')
nltk.download('punkt')
def extract_nouns(text):
    nouns = ""
    for item in pos_tag(word_tokenize(text), tagset = 'universal'):
        if item[1] == 'NOUN':
            nouns += item[0] + " "
    nouns = nouns.strip()
    return nouns

reviews_df['text'] = reviews_df['text'].apply(lambda x: extract_nouns(x))

reviews_df.to_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/reviews_df_prep_full.csv", encoding='utf-8')

## BUSINESS DATA ##

# SAVE TOP X CATEGORIES FOR ANALYSIS #
businesses_df['categories'] = businesses_df['categories'].str.split(", ")

# Batch through business dataset to save top 15 and top 30 categories
batches = [15,30]

for batch in batches:
    businesses_df_c = businesses_df.copy()
    # Dictionary to store categories, and variables to report later
    all_unique_cats = {}
    total_cats_before = businesses_df_c['categories'].apply(len).sum()
    len_before = len(businesses_df_c)

    # Iterate through each business' categories and count occurances
    for cats in businesses_df_c['categories']:
        for cat in cats:
            if cat in all_unique_cats:
                all_unique_cats[cat] += 1
            else:
                all_unique_cats[cat] = 1

    # Number of top categories to keep
    num_cats = batch
    sorted_cats = sorted(all_unique_cats.items(), key=lambda x: x[1], reverse=True)

    # Select top categories
    top_cats = [x[0] for x in sorted_cats[:num_cats]]

    # Remove reviews that belong to businesses with none of the top categories
    businesses_df_c = businesses_df_c[businesses_df_c['categories'].apply(lambda x: any(cat in top_cats for cat in x) if x else False)]

    # Remove categories from remaining reviews that are not in the top categories
    businesses_df_c['categories'] = businesses_df_c['categories'].apply(lambda x: ([i for i in x if i in top_cats]))

    # Count total number of categories
    total_cats_after = businesses_df_c['categories'].apply(len).sum()

    # Print results
    print(f"\nTotal Initial Unique Categories: {len(all_unique_cats)}")
    print(f"Number of Total Reviews: {len(businesses_df_c)}")
    print(f"Average Number of Categories Before: {total_cats_before/len_before}")
    print(f"Average Number of Categories After: {total_cats_after/len(businesses_df_c)}")

    businesses_df_c.to_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/bus_df_prep_full_" + str(batch) + ".csv", encoding='utf-8')

### MERGED DATA ###

# LOAD DATA #
reviews_df = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/reviews_df_prep_full.csv")
businesses_df_15 = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/bus_df_prep_full_15.csv")
businesses_df_30 = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/bus_df_prep_full_30.csv")

# GENERATE PACKET DF #

# Batch through each business dataset
df_businesses = [(businesses_df_15, '15'), (businesses_df_30, '30')]

for df_business, cat_amount in df_businesses:
    print(f'Categories: {cat_amount}')
    merged_df = reviews_df.merge(df_business, left_on='business_id', right_on='business_id')
    merged_df = merged_df[['business_id', 'text', 'categories']]
    print(f"Total Initial Businesses: {len(merged_df['business_id'].unique())}")

    # Track Stats
    total_bus_proc = 0
    bus_under_512 = 0

    # Iterate through each business and save 512 randomly selected tokens from all reviews
    bus_combined_dict = {'Business ID':[], 'Text':[], 'Categories':[], 'Packet Number': []}
    for bus_id in merged_df['business_id'].unique().tolist():
        # Isolate business specific information
        bus_information_df = merged_df[merged_df['business_id'] == bus_id].reset_index()
        categories = bus_information_df.loc[0,'categories']
        # Create long list of all words in review
        reviews_word_list = bus_information_df['text'].str.cat(sep=' ').split()
        # If not enough tokens to fill all packets in businesses' reviews, do not train on business
        if len(reviews_word_list) < 512:
            bus_under_512 += 1
            continue
        # Sample enough words to fill up full packets (up to 3 packets)
        if len(reviews_word_list) > 1536:
            sampled_index = random.sample(range(0, len(reviews_word_list)), 1536)
            num_full_packets = 3
        elif len(reviews_word_list) > 1024:
            sampled_index = random.sample(range(0, len(reviews_word_list)), 1024)
            num_full_packets = 2
        else:
            sampled_index = random.sample(range(0, len(reviews_word_list)), 512)
            num_full_packets = 1
        # Save each packet in a new dictionary entry and save packet number as a way to query the DF
        start_sampled_index = 0
        for i in range(1,num_full_packets+1):
            end_sampled_index = 512*i
            # Add all packets to dictionary to be saved as a DataFrame
            if len(sampled_index) >= (end_sampled_index):
                sampled_index_packet = sampled_index[start_sampled_index:end_sampled_index]
                sampled_words = [reviews_word_list[ind] for ind in sampled_index_packet]
                bus_combined_dict['Business ID'].append(bus_id)
                bus_combined_dict['Text'].append(sampled_words)
                bus_combined_dict['Categories'].append(categories)
                bus_combined_dict['Packet Number'].append(i)
            start_sampled_index = 512*i

    # Generate DataFrame
    combined_df = pd.DataFrame(bus_combined_dict)

    # Print stats
    print(f"Total Dropped Businesses: {bus_under_512}")
    print(f"Total 1+ Packet Businesses: {len(combined_df[combined_df['Packet Number'] == 1])}")
    print(f"Total 2+ Packet Businesses: {len(combined_df[combined_df['Packet Number'] == 2])}")
    print(f"Total 3+ Packet Businesses: {len(combined_df[combined_df['Packet Number'] == 3])}")

    # Batch save to csv
    combined_df.to_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/packet_df_full_" + cat_amount + ".csv", encoding='utf-8')

### TRAIN, VALIDATE, TEST SPLIT ###

# READ DATA, ISOLATE PACKET NUM, CLEAN CATEGORIES LIST #
packet_df_full_15 = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/packet_df_full_15.csv")
packet_df_full_30 = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/packet_df_full_30.csv")

# Batch through each packet dataset
df_batches = [(packet_df_full_15.copy(), "packet_df_full_15"), (packet_df_full_30.copy(), "packet_df_full_30")]

for df_batch, df_name in df_batches:
    print(f'\n\n{df_name}:')
    
    # Categories saves as a string in CSV so convert back to list
    for ch in "'[]":
        df_batch['Categories'] = df_batch['Categories'].str.replace(ch, '')
    df_batch['Categories'] = df_batch['Categories'].str.replace(", ", ",").replace(" ,", ",")
    df_batch['Categories'] = df_batch['Categories'].str.split(',')

    # Separate packet DF by packet number
    df_batch_p1 = df_batch[df_batch['Packet Number'] == 1].reset_index(drop=True)
    df_batch_p2 = df_batch[df_batch['Packet Number'] == 2].reset_index(drop=True)
    df_batch_p3 = df_batch[df_batch['Packet Number'] == 3].reset_index(drop=True)

    # Batch through each packet type
    df_packets = [(df_batch_p1, 'Packet 1'), (df_batch_p2, 'Packet 2'), (df_batch_p3, 'Packet 3')]

    for df_packet, packet_name in df_packets:
        print(packet_name)
        
        # Set up x and y values for each category number and packet type, one hot encode with MultiLabelBinarizer() from list
        x = df_packet['Text'].values
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df_packet['Categories']).astype(float)

        # Split to train / test, mark column in df
        msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
        for train_index, test_index_int in msss.split(x, y):
            x_train, x_test_temp = x[train_index], x[test_index_int]
            y_train, y_test_temp = y[train_index], y[test_index_int]

        df_packet.loc[train_index,'Type ' + packet_name] = 'Train'

        # Split to test / validation, mark columns in df
        msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=42)
        for test_index, val_index in msss.split(x_test_temp, y_test_temp):
            x_test, x_val = x_test_temp[test_index], x_test_temp[val_index]
            y_test, y_val = y_test_temp[test_index], y_test_temp[val_index]

        df_packet.loc[test_index_int[test_index],'Type ' + packet_name] = 'Test'
        df_packet.loc[test_index_int[val_index],'Type ' + packet_name] = 'Validate'

        # Test split effectiveness
        train_cats = {}
        test_cats = {}
        validate_cats = {}

        # Iterate through each business' categories and count occurances
        for ind in df_packet.index:
            cats = df_packet.loc[ind,'Categories']
            typ = df_packet.loc[ind,'Type ' + packet_name]
            for cat in cats:
                if typ == 'Train':
                    if cat in train_cats:
                        train_cats[cat] += 1
                    else:
                        train_cats[cat] = 1
                if typ == 'Test':
                    if cat in test_cats:
                        test_cats[cat] += 1
                    else:
                        test_cats[cat] = 1
                if typ == 'Validate':
                    if cat in validate_cats:
                        validate_cats[cat] += 1
                    else:
                        validate_cats[cat] = 1

        # Print split result percentages
        print("\nEVALUATION:\nKey: train (80%), validate (10%), test (10%), Total")
        for key in train_cats.keys():
            total = train_cats[key] + test_cats[key] + validate_cats[key]
            print(f"{key}: {train_cats[key]/total*100:.1f}%, {test_cats[key]/total*100:.1f}%, {validate_cats[key]/total*100:.1f}%, Total: {total}")

        # Save packet split information to overall category dataframe
        for bus_id, typ in zip(df_packet['Business ID'], df_packet['Type ' + packet_name]):
            df_batch.loc[(df_batch['Business ID'] == bus_id) & (df_batch['Packet Number'] <= int(packet_name[-1])),'Type ' + packet_name] = typ

    # Save two dataframes, one for each batch category number with all split data for all packet numbers
    df_batch.to_csv('/home/natomasetti/DS3010/Yelp_Data_Preprocessed/'+df_name+'_split.csv')

### TENSOR DATASET ###

# READ DATA, ISOLATE PACKET NUM, CLEAN CATEGORIES LIST #
packet_df_full_15_split = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/packet_df_full_15_split.csv")
packet_df_full_30_split = pd.read_csv("/home/natomasetti/DS3010/Yelp_Data_Preprocessed/packet_df_full_30_split.csv")

# Iterate through each packet df
df_batches = [(packet_df_full_15_split, "15"), (packet_df_full_30_split, "30")]

for df_batch, batch_cats in df_batches:
    print(f'\n{batch_cats} Category Packet:')
    
    # Split categories into list
    for ch in "'[]":
        df_batch['Categories'] = df_batch['Categories'].str.replace(ch, '')
    df_batch['Categories'] = df_batch['Categories'].str.replace(", ", ",").replace(" ,", ",")
    df_batch['Categories'] = df_batch['Categories'].str.split(',')
    
    # Clean text list to make a long string of words
    for ch in "'[]":
        df_batch['Text'] = df_batch['Text'].str.replace(ch, '')
    df_batch['Text'] = df_batch['Text'].str.replace(", ", ",").replace(" ,", ",")
    df_batch['Text'] = df_batch['Text'].str.split(',')
    df_batch['Text'] = df_batch['Text'].apply(lambda x: ' '.join(x))

    # Iterate through each packet count
    packets = ['Packet 1', 'Packet 2', 'Packet 3']

    for packet_name in packets:
        print(f'\n{packet_name}')

        # Generate train, validate, test DataFrames
        df_train = df_batch[df_batch['Type '+ packet_name] == "Train"]
        df_validate = df_batch[df_batch['Type '+ packet_name] == "Validate"]
        df_test = df_batch[df_batch['Type '+ packet_name] == "Test"]
        
        # Iterate through each type of split
        df_splits = [(df_test, "Test"), (df_validate, "Validate"), (df_train, "Train")]

        for df_split, split_type in df_splits:
            
            # Create inputs and outputs to encode
            inputs = df_split.Text.values
            mlb = MultiLabelBinarizer()
            outputs = mlb.fit_transform(df_split['Categories']).astype(float)
            
            # Create tokenizer, set up lists for inputs and attention masks
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            input_ids = []
            attention_masks = []
            
            # Iterate through each word and tokenize
            for rev in inputs:
                # `encode_plus` will:
                #   (1) Tokenize the sentence
                #   (2) Prepend the `[CLS]` token to the start
                #   (3) Append the `[SEP]` token to the end
                #   (4) Map tokens to their IDs
                #   (5) Pad or truncate the sentence to `max_length`
                #   (6) Create attention masks for [PAD] tokens
                encoded_dict = tokenizer.encode_plus(
                                    rev,                      # Sentence to encode
                                    add_special_tokens = False,
                                    max_length = 512,           # Pad & truncate all sentences. Reviews over 512 so truncate there for now
                                    pad_to_max_length = True,
                                    return_attention_mask = True,   # Construct attn. masks
                                    return_tensors = 'pt',     # Return pytorch tensors
                                    truncation = True        # Truncate at max_length
                            )
            
                # Add the encoded sentence to the list  
                input_ids.append(encoded_dict['input_ids'])
                
                # And its attention mask
                attention_masks.append(encoded_dict['attention_mask'])

            # Convert the lists into tensors
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            outputs = torch.tensor(outputs).clone().detach()

            # Create tensor dataset and save to file
            dataset = TensorDataset(input_ids, attention_masks, outputs)
            torch.save(dataset, '/home/natomasetti/DS3010/Yelp_Tensor_Datasets/' + batch_cats + '_' + packet_name + '/' + split_type + '_Tensor_Dataset.pt')

            # Save MultiLabelBinarizer so predictions can be converted back to labels later
            if split_type == "Test":
                with open('/home/natomasetti/DS3010/Yelp_Tensor_Datasets/' + batch_cats + '_' + packet_name + '/test_mlb.pkl', 'wb') as f:
                    pickle.dump(mlb, f)