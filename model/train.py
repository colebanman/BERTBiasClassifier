# backend/train.py
import torch
import numpy as np
from transformers import AdamW, get_cosine_schedule_with_warmup
from model import BiasClassifier
from data_preparation import LABELS
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from visualizations import plot_training_history, plot_confusion_matrix, plot_class_distribution, plot_roc_curve, plot_precision_recall_curve, plot_learning_rate
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import wordnet
import random
import pandas as pd
from torch import nn
import torch.nn.functional as F

nltk.download('wordnet', quiet=True)

# Ensure the /figs directory exists
os.makedirs('figs', exist_ok=True)

left_lexicon = ['progressive', 'liberal', 'democrat', 'socialism', 'welfare']
right_lexicon = ['conservative', 'republican', 'tradition', 'freedom', 'patriot']
neutral_lexicon = ['bipartisan', 'centrist', 'moderate', 'independent', 'neutral']

def add_lexicon_features(df):
    left_vectorizer = CountVectorizer(vocabulary=left_lexicon)
    right_vectorizer = CountVectorizer(vocabulary=right_lexicon)
    neutral_vectorizer = CountVectorizer(vocabulary=neutral_lexicon)
    
    left_counts = left_vectorizer.fit_transform(df['sentence']).toarray()
    right_counts = right_vectorizer.fit_transform(df['sentence']).toarray()
    neutral_counts = neutral_vectorizer.fit_transform(df['sentence']).toarray()
    
    df['left_lexicon_count'] = left_counts.sum(axis=1)
    df['right_lexicon_count'] = right_counts.sum(axis=1)
    df['neutral_lexicon_count'] = neutral_counts.sum(axis=1)
    
    return df

def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(set(synonyms)))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def augment_data(df, target_classes=['left', 'right', 'unbiased'], augment_factor=2):
    augmented_data = []
    for _, row in df.iterrows():
        if row['label'] in target_classes:
            for _ in range(augment_factor - 1):
                augmented_text = synonym_replacement(row['sentence'])
                augmented_data.append({
                    'sentence': augmented_text,
                    'label': row['label']
                })
    
    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_examples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        total_examples += len(labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': np.mean(losses)})
    
    return correct_predictions.double() / total_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, weight_factor=2.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_factor = weight_factor
        
    def forward(self, inputs, targets):
        weights = torch.ones(self.num_classes)
        weights[3] = self.weight_factor  # Assuming 3, 5, 6 are indices for left, right, unbiased
        weights[5] = self.weight_factor
        weights[6] = self.weight_factor
        weights = weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=weights)

def train_model(train_data_loader, val_data_loader, test_data_loader, train_data, val_data, test_data, epochs, device, output_model_path):
    # Data augmentation
    train_data = augment_data(train_data)
    
    # Feature engineering
    train_data = add_lexicon_features(train_data)
    val_data = add_lexicon_features(val_data)
    test_data = add_lexicon_features(test_data)
    
    model = BiasClassifier(n_classes=len(LABELS))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # criterion = WeightedCrossEntropyLoss(num_classes=len(LABELS))
    # Switching to CrossEntropyLoss as weighted loss is producing bad results. Iteration 3
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    best_accuracy = 0
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1}/{epochs}')
        logging.info('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler
        )

        logging.info(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss, _, _ = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device
        )

        logging.info(f'Val   loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), output_model_path)
            best_accuracy = val_acc
            logging.info(f'New best model saved with accuracy: {best_accuracy}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break

    logging.info("Training completed")
    
    # Plot training history
    logging.info("Plotting training history...")
    plot_training_history(history)
    
    # Plot learning rate schedule
    logging.info("Plotting learning rate schedule...")
    plot_learning_rate(scheduler, total_steps)
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    model.load_state_dict(torch.load(output_model_path))
    test_acc, test_loss, test_preds, test_labels = eval_model(model, test_data_loader, loss_fn, device)
    logging.info(f'Test loss {test_loss} accuracy {test_acc}')
    
    # Check if test_preds and test_labels are already NumPy arrays
    if not isinstance(test_preds, np.ndarray):
        test_preds = test_preds.cpu().numpy()
    if not isinstance(test_labels, np.ndarray):
        test_labels = test_labels.cpu().numpy()
    
    # Print unique classes
    unique_preds = np.unique(test_preds)
    unique_labels = np.unique(test_labels)
    logging.info(f"Unique predicted classes: {unique_preds}")
    logging.info(f"Unique true labels: {unique_labels}")
    
    # Get probabilities for ROC and PR curves
    if test_preds.ndim == 1:
        # If test_preds is 1D, it's likely class indices, so we need to convert to one-hot
        test_probs = np.eye(len(LABELS))[test_preds]
    else:
        # If test_preds is already 2D, we assume it contains class probabilities
        test_probs = test_preds
    
    # Plot confusion matrix
    logging.info("Plotting confusion matrix...")
    unique_classes = plot_confusion_matrix(test_labels, np.argmax(test_probs, axis=1), LABELS)
    
    # Plot class distribution
    logging.info("Plotting class distribution...")
    plot_class_distribution(train_data, val_data, test_data, LABELS, unique_classes)
    
    # Generate classification report
    logging.info("Generating classification report...")
    actual_labels = [LABELS[i] for i in unique_classes]
    report = classification_report(test_labels, np.argmax(test_probs, axis=1), target_names=actual_labels)
    logging.info("Classification Report:\n" + report)
    
    # Plot ROC curve
    logging.info("Plotting ROC curve...")
    plot_roc_curve(test_labels, test_probs, len(LABELS))
    
    # Plot Precision-Recall curve
    logging.info("Plotting Precision-Recall curve...")
    plot_precision_recall_curve(test_labels, test_probs, len(LABELS))
    
    # If you have feature importance information, uncomment the following:
    # feature_importance = ... # Calculate or retrieve feature importance
    # feature_names = ... # List of feature names
    # logging.info("Plotting top features...")
    # plot_top_features(feature_importance, feature_names)
    
    # If you want to plot attention weights, you'll need to modify your model to output attention weights
    # and then call plot_attention_weights for a sample input
    
    return history, best_accuracy, test_acc
