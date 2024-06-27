# backend/eval.py
import torch
import numpy as np
from model import BiasClassifier
from transformers import BertTokenizer
from data_preparation import LABELS, LABEL_MAP
from sklearn.metrics import confusion_matrix, classification_report

def classify_sentence(model, tokenizer, sentence, device, max_length):
    model = model.eval()
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    _, preds = torch.max(outputs, dim=1)
    return LABELS[preds.item()]

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    cm = confusion_matrix(actual_labels, predictions)
    cr = classification_report(actual_labels, predictions, target_names=LABELS)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    return cm, cr

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasClassifier(n_classes=len(LABELS))
    model.load_state_dict(torch.load('best_model_state.bin'))
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    while True:
        sentence = input("Enter a sentence (or 'q' to quit): ")
        if sentence.lower() == 'q':
            break
        prediction = classify_sentence(model, tokenizer, sentence, device, 160)
        print("Predicted Bias Category:", prediction)