# backend/visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import logging
import os

import torch


# Ensure the figs directory exists
os.makedirs('figs', exist_ok=True)

def plot_training_history(history):
    # Print the if any history values are device type tensors
    for key, value in history.items():
        if isinstance(value, torch.Tensor):
            logging.info(f"Converting {key} to numpy array")
            history[key] = value.cpu().numpy()
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')  # Use .cpu() to copy tensor to host memory
        plt.plot(history['val_loss'], label='Validation Loss')  # Use .cpu() to copy tensor to host memory
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('figs/training_history.png')
        plt.close()
        logging.info("Training history plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_training_history: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, classes):
    try:
        # Get the unique classes actually present in the data
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[classes[i] for i in unique_classes], 
                    yticklabels=[classes[i] for i in unique_classes])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('figs/confusion_matrix.png')
        plt.close()
        logging.info("Confusion matrix plot saved successfully.")
        return unique_classes
    except Exception as e:
        logging.error(f"Error in plot_confusion_matrix: {str(e)}")
        return np.unique(np.concatenate((y_true, y_pred)))

def plot_class_distribution(train_data, val_data, test_data, classes, unique_classes):
    try:
        plt.figure(figsize=(12, 6))
        
        train_dist = train_data['label'].value_counts().sort_index()
        val_dist = val_data['label'].value_counts().sort_index()
        test_dist = test_data['label'].value_counts().sort_index()
        
        x = np.arange(len(unique_classes))
        width = 0.25
        
        plt.bar(x - width, [train_dist.get(i, 0) for i in unique_classes], width, label='Train')
        plt.bar(x, [val_dist.get(i, 0) for i in unique_classes], width, label='Validation')
        plt.bar(x + width, [test_dist.get(i, 0) for i in unique_classes], width, label='Test')
        
        plt.xlabel('Classes')
        plt.ylabel('Number of samples')
        plt.title('Class Distribution in Train, Validation, and Test Sets')
        plt.xticks(x, [classes[i] for i in unique_classes], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('figs/class_distribution.png')
        plt.close()
        logging.info("Class distribution plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_class_distribution: {str(e)}")

def plot_attention_weights(attention_weights, tokens, label):
    try:
        plt.figure(figsize=(10, 10))
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(f'Attention Weights for Class: {label}')
        plt.tight_layout()
        plt.savefig(f'figs/attention_weights_{label}.png')
        plt.close()
        logging.info(f"Attention weights plot for class {label} saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_attention_weights for class {label}: {str(e)}")

def plot_learning_rate(scheduler, num_training_steps):
    try:
        lrs = []
        for step in range(num_training_steps):
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        
        plt.figure(figsize=(10, 5))
        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.tight_layout()
        plt.savefig('figs/learning_rate_schedule.png')
        plt.close()
        logging.info("Learning rate schedule plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_learning_rate: {str(e)}")

def plot_top_features(feature_importance, feature_names, top_n=20):
    try:
        sorted_idx = np.argsort(feature_importance)
        top_features = sorted_idx[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), feature_importance[top_features])
        plt.yticks(range(top_n), [feature_names[i] for i in top_features])
        plt.title(f'Top {top_n} Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('figs/top_features.png')
        plt.close()
        logging.info(f"Top {top_n} features plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_top_features: {str(e)}")

def plot_roc_curve(y_true, y_score, n_classes):
    try:
        # Binarize the output
        y_test = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('figs/roc_curve.png')
        plt.close()
        logging.info("ROC curve plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_roc_curve: {str(e)}")

def plot_precision_recall_curve(y_true, y_score, n_classes):
    try:
        # Binarize the output
        y_test = label_binarize(y_true, classes=range(n_classes))

        # Compute Precision-Recall and plot curve for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

        # Plot Precision-Recall curve for each class
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(recall[i], precision[i],
                     label=f'Precision-Recall curve of class {i} (AP = {average_precision[i]:0.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('figs/precision_recall_curve.png')
        plt.close()
        logging.info("Precision-Recall curve plot saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_precision_recall_curve: {str(e)}")