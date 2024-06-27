# backend/model.py
from transformers import BertModel
import torch.nn as nn

class BiasClassifier(nn.Module):
    def __init__(self, n_classes, freeze_bert=False):
        super(BiasClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # Extract the pooled output
        output = self.drop(pooled_output)
        return self.out(output)