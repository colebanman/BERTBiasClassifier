# backend/main.py
import torch
from transformers import BertTokenizer
from train import train_model
from data_preparation import prepare_data, create_data_loader, LABELS
from model import BiasClassifier
from eval import classify_sentence, evaluate_model
import argparse
import logging

from visualizations import plot_training_history

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Bias Classification Training")
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=160, help='Max sequence length')
    parser.add_argument('--output_model', type=str, default='best_model_state.bin', help='Output model path')
    parser.add_argument('--data_file', type=str, default='allData.json', help='Input data file path')
    return parser.parse_args()

def main(args):
    setup_logging()
    logging.info("Starting bias classification training")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load and preprocess data
    train_data, val_data, test_data = prepare_data(args.data_file)
    logging.info(f"Training data size: {len(train_data)}")
    logging.info(f"Validation data size: {len(val_data)}")
    logging.info(f"Test data size: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load and prepare data
    train_data, val_data, test_data = prepare_data(args.data_file)
    
    # Create data loaders
    train_data_loader = create_data_loader(train_data, tokenizer, args.max_length, args.batch_size)
    val_data_loader = create_data_loader(val_data, tokenizer, args.max_length, args.batch_size)
    test_data_loader = create_data_loader(test_data, tokenizer, args.max_length, args.batch_size)
    
    # Train the model
    history, best_accuracy, test_accuracy = train_model(
        train_data_loader, 
        val_data_loader,
        test_data_loader,
        train_data,
        val_data,
        test_data,
        args.epochs,
        device,
        args.output_model
    )
    
    logging.info(f"Best validation accuracy: {best_accuracy}")

    # Load the trained model
    model = BiasClassifier(n_classes=len(LABELS))
    model.load_state_dict(torch.load(args.output_model))
    model = model.to(device)

    # Evaluate the model on the test set
    logging.info("Evaluating model on test set")
    evaluate_model(model, test_data_loader, device)

    # Classify a sample sentence
    sentence = "This is an example biased sentence."
    prediction = classify_sentence(model, tokenizer, sentence, device, args.max_length)
    logging.info(f"Sample sentence: '{sentence}'")
    logging.info(f"Predicted Bias Category: {prediction}")

if __name__ == "__main__":
    args = parse_args()
    main(args)