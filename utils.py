# utils.py
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure GPU memory growth to avoid memory issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.warning(f"GPU memory growth configuration failed: {e}")

class TextProcessor:
    def __init__(self, max_vocab_size: int = 50000, min_word_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freqs = {}
    
    def fit(self, texts: List[str]) -> None:
        # First pass: count all words
        for text in texts:
            # Improved tokenization
            words = text.lower().split()  # Convert to lowercase
            for word in words:
                self.word_freqs[word] = self.word_freqs.get(word, 0) + 1
        
        # Filter words by frequency and vocab size
        filtered_words = [
            (word, count) for word, count in self.word_freqs.items()
            if count >= self.min_word_freq
        ]
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        filtered_words = filtered_words[:self.max_vocab_size]
        
        # Create vocabulary mappings
        for idx, (word, _) in enumerate(filtered_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        logging.info(f"Vocabulary size: {self.vocab_size}")
        logging.info(f"Total unique words: {len(self.word_freqs)}")
        logging.info(f"Words included in vocabulary: {(self.vocab_size/len(self.word_freqs))*100:.2f}%")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            words = text.lower().split()  # Convert to lowercase
            # Handle out-of-vocabulary words by finding closest match
            sequence = []
            for word in words:
                if word in self.word2idx:
                    sequence.append(self.word2idx[word])
                else:
                    # Try to find the closest word in vocabulary
                    closest_word = self._find_closest_word(word)
                    sequence.append(self.word2idx[closest_word])
            sequences.append(sequence)
        return sequences
    
    def _find_closest_word(self, word: str) -> str:
        """Find the closest word in vocabulary using basic string similarity."""
        if not self.word2idx:
            return list(self.word2idx.keys())[0]  # Return first word if no match
            
        # Simple character-based similarity
        def similarity(w1: str, w2: str) -> float:
            shorter = min(len(w1), len(w2))
            return sum(c1 == c2 for c1, c2 in zip(w1[:shorter], w2[:shorter])) / shorter
        
        best_match = max(self.word2idx.keys(), 
                        key=lambda x: similarity(word, x))
        return best_match
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'word_freqs': self.word_freqs
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TextProcessor':
        processor = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            processor.word2idx = data['word2idx']
            processor.idx2word = data['idx2word']
            processor.vocab_size = data['vocab_size']
            processor.word_freqs = data.get('word_freqs', {})
        return processor

class WordRNN(Model):
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 256,
                 rnn_units: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        self.rnn_layers = [
            layers.LSTM(rnn_units, 
                       return_sequences=True, 
                       dropout=dropout) 
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout)
        self.dense = layers.Dense(vocab_size)
        
        # Store sequence length for generation
        self.seq_length = None
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, training=training)
        
        x = self.dropout(x, training=training)
        return self.dense(x)
    
    def generate_next_word(self, input_sequence, temperature=1.0):
        predictions = self(input_sequence)
        predictions = predictions[0, -1, :] / temperature
        return tf.random.categorical(predictions[None, :], 1)[0, 0]
    
    def set_sequence_length(self, length: int):
        self.seq_length = length