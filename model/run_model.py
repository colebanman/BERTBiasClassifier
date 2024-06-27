from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import BiasClassifier
from data_preparation import LABELS, preprocess_text
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiasClassifier(n_classes=len(LABELS))
model.load_state_dict(torch.load('backend/best_model_state.bin', map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def classify_sentence(sentence, max_length=160):
    logging.info(f"Classifying sentence: {sentence}")
    preprocessed_sentence = preprocess_text(sentence)
    encoding = tokenizer.encode_plus(
        preprocessed_sentence,
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    probabilities = F.softmax(outputs, dim=1)
    confidence, preds = torch.max(probabilities, dim=1)
    bias_type = LABELS[preds.item()]
    confidence_value = confidence.item() * 100
    
    logging.info(f"Classification result: {bias_type} with confidence {confidence_value:.2f}%")
    return bias_type, confidence_value

def get_suggestion(bias_type):
    classes = ['class', 'confirmation', 'cultural', 'left',
         'racial', 'right', 'unbiased', 'age']
    suggestions = {
        'unbiased': 'The sentence appears to be unbiased.',
        'confirmation': 'This sentence may reinforce pre-existing beliefs. Consider this when reading.',
        'left': 'This sentence leans towards left-wing political perspectives.',
        'right': 'This sentence leans towards right-wing political perspectives.',
        'cultural': 'This sentence reflects biases about a specific culture\'s traditions, beliefs, or practices.',
        'racial': 'This sentence expresses prejudice against individuals or groups. Please be cautious when reading.',
        'age': 'This sentence displays bias towards a particular age group, often through stereotypes.',
        'class': 'This sentence demonstrates prejudice based on socioeconomic status.'
    }
    return suggestions.get(bias_type, 'Consider revising the sentence to reduce potential bias.')

@app.route('/analyze_bulk', methods=['POST'])
def analyze_bulk():
    data = request.json
    sentences = data['sentences']
    logging.info(f"Received {len(sentences)} sentences for analysis")
    
    results = []
    for sentence in sentences:
        bias_type, confidence = classify_sentence(sentence)
        suggestion = get_suggestion(bias_type)
        
        result = {
            'biased': bias_type != 'unbiased',
            'type': bias_type,
            'confidence': confidence,
            'suggestion': suggestion
        }
        results.append(result)
        logging.info(f"Sentence analysis result: {result}")
    
    logging.info(f"Completed analysis of {len(sentences)} sentences")
    return jsonify(results)

if __name__ == '__main__':
    logging.info("Starting Bias Detection Server")
    app.run(debug=True)