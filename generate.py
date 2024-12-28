# generate.py
import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from typing import List
from utils import TextProcessor, WordRNN

def generate_text(model: WordRNN,
                 processor: TextProcessor,
                 seed_text: str,
                 num_words: int = 100,
                 temperature: float = 1.0,
                 stop_tokens: List[str] = None) -> str:
    """Generate text using the trained model."""
    seed_text = seed_text.lower()
    
    # Convert seed text to sequence
    seed_sequence = processor.texts_to_sequences([seed_text])[0]
    generated_text = seed_text.split()

    # Ensure seed sequence is the right length
    if len(seed_sequence) < model.seq_length:
        padding = [0] * (model.seq_length - len(seed_sequence))
        seed_sequence = padding + seed_sequence
    elif len(seed_sequence) > model.seq_length:
        seed_sequence = seed_sequence[-model.seq_length:]

    # Main generation loop
    for _ in range(num_words):
            
        # Prepare input sequence
        input_seq = np.array([seed_sequence])

        # Generate prediction with temperature scaling
        predicted_id = model.generate_next_word(input_seq, temperature)

        # Convert predicted ID to word
        predicted_word = processor.idx2word.get(predicted_id.numpy(), '')
        if predicted_word:
            generated_text.append(predicted_word)

        # Update seed sequence
        seed_sequence = np.append(seed_sequence[1:], predicted_id)

        # Check for stop tokens
        if stop_tokens and predicted_word in stop_tokens:
            break

    return ' '.join(generated_text)

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained WordRNN model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--epoch', type=int, help='Specific epoch checkpoint to use (e.g., 10 for epoch 10)')
    parser.add_argument('--num_words', type=int, default=100, help='Number of words to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--seed_text', type=str, help='Seed text for generation')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--rnn_units', type=int, default=512, help='Number of RNN units')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length')

    args = parser.parse_args()

    # Clamp temperature
    temperature = max(0.1, min(args.temperature, 10.0))

    # Load processor
    processor_path = os.path.join(args.model_dir, 'processor.pkl')
    if not os.path.exists(processor_path):
        logging.error("Text processor file not found.")
        return
    processor = TextProcessor.load(processor_path)

    # Create and configure model
    model = WordRNN(
        vocab_size=processor.vocab_size,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers
    )

    # Set sequence length
    model.set_sequence_length(args.seq_length)

    # Build the model with dummy input
    dummy_input = tf.zeros((1, args.seq_length))
    _ = model(dummy_input)

    # Determine which checkpoint to load
    if args.epoch is not None:
        # Load specific epoch checkpoint
        checkpoint_path = os.path.join(args.model_dir, f'epoch_{args.epoch}_weights.weights.h5')
        if not os.path.exists(checkpoint_path):
            available_checkpoints = [f for f in os.listdir(args.model_dir) if f.endswith('.weights.h5')]
            print(f"Checkpoint for epoch {args.epoch} not found.")
            print(f"Available checkpoints: {available_checkpoints}")
            return
    else:
        # Try to load the latest checkpoint
        checkpoints = [f for f in os.listdir(args.model_dir) if f.endswith('.weights.h5')]
        if not checkpoints:
            print("No checkpoints found in directory.")
            return
        
        # Find the latest checkpoint by sorting epochs
        epochs = [int(f.split('_')[1]) for f in checkpoints if f.startswith('epoch_')]
        latest_epoch = max(epochs)
        checkpoint_path = os.path.join(args.model_dir, f'epoch_{latest_epoch}_weights.weights.h5')
        print(f"Using checkpoint from epoch {latest_epoch}")

    # Load weights
    try:
        model.load_weights(checkpoint_path)
        logging.info(f"Loaded weights from: {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        return

    # Get seed text
    seed_text = args.seed_text
    if seed_text is None:
        seed_text = input("Enter seed text: ")

    # Define stop tokens for natural text endings
    stop_tokens = {'.', '!', '?'}

    # Generate text
    generated_text = generate_text(
        model,
        processor,
        seed_text,
        num_words=args.num_words,
        temperature=temperature,
        stop_tokens=stop_tokens
    )

    print("\nGenerated text:\n")
    print(generated_text)

if __name__ == '__main__':
    main()