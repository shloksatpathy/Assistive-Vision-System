import json
import torch
from collections import Counter

class Vocabulary:
    """
    Constructs a vocabulary from a corpus of captions.
    Allows mapping words to integer indices (numericalize) and back (for inference).
    """
    def __init__(self, freq_threshold=2):
        # We define a few special tokens commonly used in NLP tasks
        # <pad> for padding variable-length sequences
        # <start> to tell the decoder when to begin
        # <end> to tell the decoder when to stop
        # <unk> for out-of-vocabulary words
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer(text):
        """
        Simple tokenizer. Can be replaced with something more complex (NLTK/SpaCy) if needed.
        Currently just splits by space.
        """
        return text.strip().lower().split()
    
    def build_vocabulary(self, sentence_list):
        """
        Builds the vocab from a list of sentences.
        """
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                
                # Add word to vocab if it hits the frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text):
        """
        Converts a text string to a list of integer indices based on the built vocab.
        """
        tokenized_text = self.tokenizer(text)
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]
        
    def save_vocabulary(self, file_path):
        """Saves the vocabulary state dict"""
        torch.save({
            'itos': self.itos,
            'stoi': self.stoi,
            'freq_threshold': self.freq_threshold
        }, file_path)
        
    @classmethod
    def load_vocabulary(cls, file_path):
        """Loads a vocabulary from a saved file"""
        checkpoint = torch.load(file_path, weights_only=True)
        vocab = cls(freq_threshold=checkpoint['freq_threshold'])
        vocab.itos = checkpoint['itos']
        vocab.stoi = checkpoint['stoi']
        return vocab

def build_and_save_vocab(captions_file, vocab_file, freq_threshold=2):
    """
    Reads the captions JSON, builds the vocabulary, and saves it.
    """
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)
        
    all_captions = []
    for caps in captions_dict.values():
        all_captions.extend(caps)
        
    print(f"Total captions found: {len(all_captions)}")
    print(f"Building vocabulary (frequency threshold={freq_threshold})...")
    
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(all_captions)
    
    print(f"Vocabulary built! Total unique words: {len(vocab)}")
    
    vocab.save_vocabulary(vocab_file)
    print(f"Vocabulary saved to {vocab_file}")
    
    return vocab

if __name__ == "__main__":
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent 
    CAPTIONS_FILE = BASE_DIR / "dataset" / "Flickr8K" / "clean_captions.txt"
    VOCAB_FILE = BASE_DIR / "dataset" / "Flickr8K" / "vocab.pth"
    
    if CAPTIONS_FILE.exists():
        build_and_save_vocab(CAPTIONS_FILE, VOCAB_FILE)
    else:
        print(f"Captions file not found at {CAPTIONS_FILE}")
