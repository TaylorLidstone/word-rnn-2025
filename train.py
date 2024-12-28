# train.py
import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from typing import Tuple
from utils import TextProcessor, WordRNN
logging.basicConfig(level=logging.DEBUG)

def get_memory_usage():
    """Get current memory usage of the process"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def load_checkpoint_if_available(model: WordRNN, checkpoint_dir: str) -> int:
    """Load the most recent checkpoint and return the starting epoch."""
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')
    ]
    if not checkpoint_files:
        logging.info("No checkpoints found, starting from scratch.")
        return 0

    # Sort checkpoints to find the most recent one
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    logging.info(f"Loading weights from: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

    # Extract epoch number from the filename
    starting_epoch = int(checkpoint_files[-1].split('_')[1])
    return starting_epoch
    
print(f"Current memory usage: {get_memory_usage():.2f} MB")

def create_training_sequences(text: str, 
                            seq_length: int,
                            processor: TextProcessor) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-target pairs for training."""
    print(f"Converting text to sequences...")
    # Convert text to sequence of indices
    sequence = processor.texts_to_sequences([text])[0]
    print(f"Created sequence of length: {len(sequence)}")
    
    if len(sequence) <= seq_length:
        raise ValueError(f"Text sequence length ({len(sequence)}) must be greater than seq_length ({seq_length})")
    
    # Create input-target pairs
    print(f"Creating input-target pairs...")
    input_sequences = []
    target_sequences = []
    
    total_sequences = len(sequence) - seq_length
    print(f"Will create {total_sequences} sequences")
    
    # Add progress reporting
    report_every = total_sequences // 10  # Report progress 10 times
    
    for i in range(0, len(sequence) - seq_length):
        if i % report_every == 0:
            print(f"Processing sequence {i}/{total_sequences}...")
        input_sequences.append(sequence[i:i + seq_length])
        target_sequences.append(sequence[i + 1:i + seq_length + 1])
    
    print(f"Converting to numpy arrays...")
    input_array = np.array(input_sequences)
    target_array = np.array(target_sequences)
    
    print(f"Final shapes - Input: {input_array.shape}, Target: {target_array.shape}")
    return input_array, target_array

def train_model(model: WordRNN,
                train_data: Tuple[np.ndarray, np.ndarray],
                batch_size: int = 64,
                epochs: int = 50,
                learning_rate: float = 0.001,
                checkpoint_dir: str = 'checkpoints',
                starting_epoch: int = 0) -> None:
    """Train the model with the given data and save periodic checkpoints."""
    
    print(f"Training configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Starting epoch: {starting_epoch}")
    print(f"- Training data shapes: {train_data[0].shape}, {train_data[1].shape}")
    
    # Add learning rate scheduling
    initial_learning_rate = learning_rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        # Add gradient clipping
        #gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    x_train, y_train = train_data
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    
    # Calculate steps per epoch
    steps_per_epoch = len(x_train) // batch_size
    print(f"Starting training with {steps_per_epoch} steps per epoch")
    
    for epoch in range(starting_epoch, epochs):
        print(f"Starting epoch {epoch + 1}")
        total_loss = 0
        batch_count = 0
        
        try:
            dataset_iter = iter(dataset)
            
            for batch in range(steps_per_epoch):
                try:
                    x, y = next(dataset_iter)
                    loss = train_step(x, y)
                    total_loss += loss
                    batch_count += 1
                    
                    if batch % 100 == 0:
                        logging.info(f'Epoch {epoch + 1}, Batch {batch}, Loss: {loss:.4f}')
                
                except StopIteration:
                    break
            
            # Calculate and log average loss for the epoch
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                logging.info(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
                
                # Save checkpoint after each epoch with a unique filename
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_weights.weights.h5')
                
                print(f"Attempting to save checkpoint at: {checkpoint_path}")
                model.save_weights(checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")
            
        except tf.errors.OutOfRangeError:
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                logging.info(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f} (early end)')
            continue

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'final_weights.weights.h5')
    logging.info(f"Saving final weights to: {final_path}")
    model.save_weights(final_path)
    logging.info("Saved final checkpoint")

def main():
    parser = argparse.ArgumentParser(description='Train WordRNN model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input text file')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='Directory to save model')
    parser.add_argument('--seq_length', type=int, default=50, help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--rnn_units', type=int, default=512, help='Number of RNN units')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
  # Load and preprocess data
    print("Loading input file...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded text with {len(text)} characters and approximately {len(text.split())} words")
    
    # Create and fit text processor
    print("Creating and fitting text processor...")
    processor = TextProcessor(max_vocab_size=50000, min_word_freq=2)
    processor.fit([text])
    processor.save(os.path.join(args.model_dir, 'processor.pkl'))
    
    # Create model
    print("Creating model...")
    model = WordRNN(
        vocab_size=processor.vocab_size,
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers
    )
    model.set_sequence_length(args.seq_length)
    
    # Build the model with dummy input
    print("Building model...")
    dummy_input = tf.zeros((1, args.seq_length))
    _ = model(dummy_input)

    # Load checkpoint if available
    starting_epoch = load_checkpoint_if_available(model, args.model_dir)

    # Prepare training data
    print("Creating training sequences...")
    train_data = create_training_sequences(text, args.seq_length, processor)
    print(f"Created training sequences with shapes: {train_data[0].shape}, {train_data[1].shape}")

    # Train the model
    print("Starting training...")
    train_model(
        model,
        train_data,
        args.batch_size,
        args.epochs,
        checkpoint_dir=args.model_dir,
        starting_epoch=starting_epoch
    )
    
    logging.info("Training completed")

if __name__ == '__main__':
    main()