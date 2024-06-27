# backend/bench.py
import torch
from transformers import BertTokenizer
from model import BiasClassifier
from train import train_model
from data_preparation import load_data

if __name__ == "__main__":
    train_data = load_data('data_train.json')
    val_data = load_data('data_val.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10
    batch_size = 16
    max_length = 160
    output_model_path = 'best_model_state.bin'

    history, best_accuracy = train_model(
        train_data, val_data, epochs, batch_size, max_length, device, output_model_path
    )

    print(f"Best validation accuracy: {best_accuracy}")