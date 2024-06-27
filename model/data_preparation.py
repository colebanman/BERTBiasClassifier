# backend/data_preparation.py
import json
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

LABELS = ['class', 'confirmation', 'cultural', 'left',
         'racial', 'right', 'unbiased', 'age']
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"The file {file_path} is not a valid JSON file.")

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def augment_data(df):
    # Simple data augmentation: duplicate minority classes
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    
    augmented_data = []
    for label, count in class_counts.items():
        class_data = df[df['label'] == label]
        if count < max_count:
            augmented_data.append(class_data.sample(max_count - count, replace=True))
    
    return pd.concat([df] + augmented_data, ignore_index=True)

class BiasDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['sentence']
        label = self.data.iloc[idx]['label']
        preprocessed_sentence = preprocess_text(sentence)
        encoding = self.tokenizer.encode_plus(
            preprocessed_sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(LABEL_MAP[label], dtype=torch.long)
        }

def create_data_loader(data, tokenizer, max_length, batch_size):
    ds = BiasDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def prepare_data(file_path, test_size=0.2, val_size=0.1):
    df = load_data(file_path)
    df = augment_data(df)

    # Check for any NaN values in the dataset, log and remove them. Print size of dataset before and after
    logging.info(f"Number of sentences: {len(df)}")
    df = df.dropna()
    logging.info(f"Number of sentences after removing NaN values: {len(df)}")
    
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'])
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=train_val['label'])

    # What percentage of the data is used for training, validation, and testing?
    # Training: 70%, Validation: 10%, Testing: 20%

    return train, val, test