"""
Character-level tokenizer for Mini GPT
Simple but effective for small datasets like TinyShakespeare
"""

class CharTokenizer:
    """
    Character-level tokenizer that maps each unique character to an integer ID.
    
    Why character-level?
    - Simple to implement and understand
    - No out-of-vocabulary issues
    - Works well for small datasets
    - Good for learning fundamentals before moving to BPE/WordPiece
    """
    
    def __init__(self, text=None):
        """
        Initialize tokenizer with optional training text.
        
        Args:
            text (str): Text corpus to build vocabulary from
        """
        self.vocab = {}  # char -> id
        self.inverse_vocab = {}  # id -> char
        
        if text is not None:
            self.build_vocab(text)
    
    def build_vocab(self, text):
        """
        Build vocabulary from text by collecting all unique characters.
        
        Args:
            text (str): Training text corpus
        """
        # Get all unique characters and sort them for reproducibility
        unique_chars = sorted(list(set(text)))
        
        # Create bidirectional mappings
        self.vocab = {ch: i for i, ch in enumerate(unique_chars)}
        self.inverse_vocab = {i: ch for i, ch in enumerate(unique_chars)}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Characters: {''.join(unique_chars[:50])}...")  # Show first 50
    
    def encode(self, text):
        """
        Convert text to list of integer IDs.
        
        Args:
            text (str): Text to encode
            
        Returns:
            list[int]: List of token IDs
        """
        return [self.vocab[ch] for ch in text]
    
    def decode(self, ids):
        """
        Convert list of integer IDs back to text.
        
        Args:
            ids (list[int]): List of token IDs
            
        Returns:
            str: Decoded text
        """
        return ''.join([self.inverse_vocab[i] for i in ids])
    
    @property
    def vocab_size(self):
        """Return vocabulary size"""
        return len(self.vocab)
    
    def save(self, path):
        """Save vocabulary to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
    
    def load(self, path):
        """Load vocabulary from file"""
        import json
        with open(path, 'r') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


# Optional: BPE Tokenizer stub for future enhancement
class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer for more efficient encoding.
    
    Why BPE?
    - Better for larger vocabularies
    - More efficient than character-level
    - Used by GPT-2, GPT-3
    
    This is a placeholder for future implementation.
    For interviews, you can explain:
    1. BPE iteratively merges most frequent character pairs
    2. Creates subword units (between char and word level)
    3. Balances vocabulary size and sequence length
    """
    
    def __init__(self):
        raise NotImplementedError(
            "BPE tokenizer is left as an exercise. "
            "Key idea: iteratively merge most frequent byte pairs. "
            "See https://arxiv.org/abs/1508.07909"
        )
