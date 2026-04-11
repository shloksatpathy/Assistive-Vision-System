import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for Image Caption Generation.
    Takes image features and caption sequences, and outputs vocabulary scores.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_seq_length=20):
        """
        Initializes the LSTM Decoder.
        
        Args:
            embed_size (int): Size of the word embeddings and image features.
            hidden_size (int): Number of features in the LSTM hidden state.
            vocab_size (int): Total number of unique words in the vocabulary.
            num_layers (int): Number of recurrent layers.
            max_seq_length (int): Maximum length of generated captions.
        """
        super(LSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Word embedding layer that maps words to a vector of `embed_size`
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer taking the embeddings (and initially the image feature) as input
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Linear layer mapping the LSTM hidden states to the vocabulary space
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        Forward pass for training.
        
        Args:
            features (torch.Tensor): Image features from the CNN encoder. Shape: (batch_size, embed_size)
            captions (torch.Tensor): Tokenized ground-truth captions. Shape: (batch_size, seq_length)
            
        Returns:
            torch.Tensor: Outputs from the linear layer. Shape: (batch_size, seq_length, vocab_size)
        """
        # Remove the <end> token from the captions effectively, 
        # so we predict the sequence including the <end> token based on image + everything up to <end>
        # The captions input here is assumed to have both <start> and <end> tokens.
        # We don't want to pass the <end> token as an input to predict the next word.
        # So we slice: captions[:, :-1]
        embeddings = self.embed(captions[:, :-1])
        
        # The image features are treated as the very first "word" in the sequence.
        # We concatenate the features along the sequence dimension (dim=1).
        # features shape after unsqueeze: (batch_size, 1, embed_size)
        # embeddings shape: (batch_size, seq_length - 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # Pass embeddings through the LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Pass the LSTM output through the linear layer to get predictions over vocabulary
        outputs = self.linear(lstm_out)
        
        return outputs
    
    def generate_caption(self, features, vocab, max_length=None):
        """
        Generates a caption word-by-word given an image feature (for inference/testing).
        
        Args:
            features (torch.Tensor): Image features from CNN of shape (1, embed_size).
            vocab (object): Vocabulary mapping indices to words (must have .stoi or a way to get word string).
            max_length (int): Maximum length to generate.
            
        Returns:
            list: A list of predicted word indices.
        """
        if max_length is None:
            max_length = self.max_seq_length
            
        predicted_indices = []
        
        # We feed the CNN features as the initial input to the LSTM
        inputs = features.unsqueeze(1) # shape: (1, 1, embed_size)
        states = None # LSTM hidden and cell states (initialize as zeros internally)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Pass current input and states to LSTM
                lstm_out, states = self.lstm(inputs, states)
                
                # Pass output to linear layer
                outputs = self.linear(lstm_out.squeeze(1)) # shape: (1, vocab_size)
                
                # Get the word with the highest probability
                predicted = outputs.argmax(1) # shape: (1)
                
                predicted_indices.append(predicted.item())
                
                # If we predict the <end> token, we stop generating.
                # Assuming vocab has e.g. vocab.stoi['<end>']
                # (You will need to pass the actual ID of '<end>' into the generation logic realistically)
                # But for now, we'll just check if it generates something we know acts as end.
                # Since the vocab object isn't fully defined yet, we'll keep generating till max_length 
                # or until you add an explicit check for the <end> token index here.
                
                # The predicted word becomes the input for the next time step
                inputs = self.embed(predicted).unsqueeze(1) # shape: (1, 1, embed_size)
                
        return predicted_indices

    def generate_caption_beam_search(self, features, vocab, beam_size=3, max_length=None):
        """
        Generates a caption using beam search.
        """
        import torch.nn.functional as F
        if max_length is None:
            max_length = self.max_seq_length
            
        end_idx = vocab.stoi.get("<end>", 2) if vocab and hasattr(vocab, 'stoi') else 2
        k = beam_size
        
        inputs = features.unsqueeze(1)
        lstm_out, states = self.lstm(inputs, None)
        outputs = self.linear(lstm_out.squeeze(1))
        log_probs = F.log_softmax(outputs, dim=1)
        
        top_k_log_probs, top_k_words = log_probs.topk(k, dim=1)
        
        sequences = [[word.item()] for word in top_k_words[0]]
        top_k_scores = top_k_log_probs[0]
        
        h, c = states
        h = h.expand(-1, k, -1).contiguous()
        c = c.expand(-1, k, -1).contiguous()
        states = (h, c)
        
        inputs = self.embed(top_k_words[0]).unsqueeze(1)
        
        completed_seqs = []
        
        with torch.no_grad():
            for step in range(max_length - 1):
                if k == 0:
                    break
                    
                lstm_out, states = self.lstm(inputs, states)
                outputs = self.linear(lstm_out.squeeze(1))
                log_probs = F.log_softmax(outputs, dim=1)
                
                log_probs = log_probs + top_k_scores.unsqueeze(1)
                log_probs = log_probs.view(-1)
                
                top_k_scores, top_k_indices = log_probs.topk(k, dim=0)
                
                prev_word_beams = top_k_indices // self.vocab_size
                next_words = top_k_indices % self.vocab_size
                
                new_sequences = []
                active_h = []
                active_c = []
                active_scores = []
                active_words = []
                
                for i in range(k):
                    beam = prev_word_beams[i].item()
                    word = next_words[i].item()
                    score = top_k_scores[i].item()
                    
                    seq = sequences[beam] + [word]
                    
                    if word == end_idx:
                        completed_seqs.append((seq, score / len(seq)))
                    else:
                        new_sequences.append(seq)
                        active_scores.append(score)
                        active_words.append(word)
                        active_h.append(states[0][:, beam, :])
                        active_c.append(states[1][:, beam, :])
                        
                k = len(new_sequences)
                if k == 0:
                    break
                    
                sequences = new_sequences
                top_k_scores = torch.tensor(active_scores, device=features.device)
                inputs = self.embed(torch.tensor(active_words, device=features.device)).unsqueeze(1)
                states = (torch.stack(active_h, dim=1), torch.stack(active_c, dim=1))
                
        if len(completed_seqs) == 0:
            for i in range(k):
                completed_seqs.append((sequences[i], top_k_scores[i].item() / len(sequences[i])))
                
        completed_seqs.sort(key=lambda x: x[1], reverse=True)
        return completed_seqs[0][0]

    def generate_captions_top_k(self, features, vocab, beam_size=5, max_length=None, top_k=5):
        """
        Generates multiple caption candidates using beam search and returns the top-K results.
        
        Args:
            features (torch.Tensor): Image features from CNN of shape (1, embed_size).
            vocab (object): Vocabulary object with stoi/itos mappings.
            beam_size (int): Width of the beam search.
            max_length (int): Maximum caption length.
            top_k (int): Number of top candidates to return.
            
        Returns:
            list of tuple: Each element is (word_indices_list, normalized_log_prob_score).
        """
        import torch.nn.functional as F
        if max_length is None:
            max_length = self.max_seq_length
            
        end_idx = vocab.stoi.get("<end>", 2) if vocab and hasattr(vocab, 'stoi') else 2
        k = beam_size
        
        # Initial step: feed image features
        inputs = features.unsqueeze(1)
        lstm_out, states = self.lstm(inputs, None)
        outputs = self.linear(lstm_out.squeeze(1))
        log_probs = F.log_softmax(outputs, dim=1)
        
        top_k_log_probs, top_k_words = log_probs.topk(k, dim=1)
        
        sequences = [[word.item()] for word in top_k_words[0]]
        top_k_scores = top_k_log_probs[0]
        
        h, c = states
        h = h.expand(-1, k, -1).contiguous()
        c = c.expand(-1, k, -1).contiguous()
        states = (h, c)
        
        inputs = self.embed(top_k_words[0]).unsqueeze(1)
        
        completed_seqs = []
        
        with torch.no_grad():
            for step in range(max_length - 1):
                if k == 0:
                    break
                    
                lstm_out, states = self.lstm(inputs, states)
                outputs = self.linear(lstm_out.squeeze(1))
                log_probs = F.log_softmax(outputs, dim=1)
                
                log_probs = log_probs + top_k_scores.unsqueeze(1)
                log_probs = log_probs.view(-1)
                
                top_k_scores, top_k_indices = log_probs.topk(k, dim=0)
                
                prev_word_beams = top_k_indices // self.vocab_size
                next_words = top_k_indices % self.vocab_size
                
                new_sequences = []
                active_h = []
                active_c = []
                active_scores = []
                active_words = []
                
                for i in range(k):
                    beam = prev_word_beams[i].item()
                    word = next_words[i].item()
                    score = top_k_scores[i].item()
                    
                    seq = sequences[beam] + [word]
                    
                    if word == end_idx:
                        completed_seqs.append((seq, score / len(seq)))
                    else:
                        new_sequences.append(seq)
                        active_scores.append(score)
                        active_words.append(word)
                        active_h.append(states[0][:, beam, :])
                        active_c.append(states[1][:, beam, :])
                        
                k = len(new_sequences)
                if k == 0:
                    break
                    
                sequences = new_sequences
                top_k_scores = torch.tensor(active_scores, device=features.device)
                inputs = self.embed(torch.tensor(active_words, device=features.device)).unsqueeze(1)
                states = (torch.stack(active_h, dim=1), torch.stack(active_c, dim=1))
                
        # Add any remaining incomplete sequences
        if len(completed_seqs) == 0:
            for i in range(len(sequences)):
                completed_seqs.append((sequences[i], top_k_scores[i].item() / len(sequences[i])))
        
        # Sort by normalized score and return top_k
        completed_seqs.sort(key=lambda x: x[1], reverse=True)
        return completed_seqs[:top_k]

if __name__ == "__main__":
    # Test the LSTM Decoder
    BATCH_SIZE = 4
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 10000 
    SEQ_LENGTH = 15
    
    decoder = LSTMDecoder(
        embed_size=EMBED_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        vocab_size=VOCAB_SIZE
    )
    
    # Dummy image features coming from the CNN encoder
    dummy_features = torch.randn(BATCH_SIZE, EMBED_SIZE)
    
    # Dummy tokenized captions (represented by integer vocabulary indices)
    dummy_captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    
    print(f"\nDummy Image Features Shape: {dummy_features.shape}")
    print(f"Dummy Captions Shape: {dummy_captions.shape}")
    
    # Forward Pass
    outputs = decoder(dummy_features, dummy_captions)
    
    print(f"\nOutput Shape: {outputs.shape}")
    print(f"Expected Output Shape: ({BATCH_SIZE}, {SEQ_LENGTH}, {VOCAB_SIZE})")
    
    if outputs.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE):
        print("\nSUCCESS: LSTM Decoder training forward pass is working as expected!")
    else:
        print("\nERROR: Output shape does not match expectation.")
        
    print("\nSimulating Inference Generation...")
    single_feature = dummy_features[0].unsqueeze(0)
    # the method generate_caption expects a feature of shape (1, embed_size)
    word_indices = decoder.generate_caption(features=single_feature, vocab=None, max_length=10)
    print(f"Generated Indices (length {len(word_indices)}): {word_indices}")
