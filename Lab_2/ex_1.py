from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.merges = [] 
        self.token_to_id = {}
        self.id_to_token = {}
    
    def normalize(self, text):
        return text.lower().strip()
    
    def pretokenize(self, text):
        return text.split()
    
    def get_stats(self, corpus_tokens):
        pairs = defaultdict(int)
        for word, freq in corpus_tokens.items():
            symbols = word.split()  
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += freq
        return pairs
    
    def merge_pair(self, pair, corpus_tokens):
        big = pair[0] + pair[1]
        new_corpus = {}

        for word, freq in corpus_tokens.items():
            symbols = word.split()
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair:
                    new_symbols.append(big)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word = " ".join(new_symbols)
            if new_word in new_corpus:
                new_corpus[new_word] += freq
            else:
                new_corpus[new_word] = freq
        return new_corpus
    
    def train(self, corpus_sentences):
        corpus_tokens = dict()

        for sent in corpus_sentences:
            norm = self.normalize(sent)
            for word in self.pretokenize(norm):
                char_seq = ""
                for ch in word:
                    char_seq += ch + " "

                if char_seq in corpus_tokens:
                    corpus_tokens[char_seq] += 1
                else:
                    corpus_tokens[char_seq] = 1
        
        tokens = set()
        for w in corpus_tokens:
            for ch in w.split():
                tokens.add(ch)

        while len(tokens) < self.vocab_size:
            pairs = self.get_stats(corpus_tokens)   # frecventa perechilor de simboluri
            if not pairs:
                break
            best_pair, best_freq = max(pairs.items(), key=lambda kv: kv[1])

            if best_freq < 2:
                break

            self.merges.append(best_pair)
            corpus_tokens = self.merge_pair(best_pair, corpus_tokens)   
            tokens.add(best_pair[0] + best_pair[1])

        final_tokens = sorted(tokens)
        self.token_to_id = {t: i for i, t in enumerate(final_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
    
    def encode_word(self, word):
        symbols = list(word)
        merges = {a + b: (a, b) for (a, b) in self.merges}
        symbols = symbols[:]  
        while True:
            made_merge = False
            j = 0
            new_symbols = []
            while j < len(symbols):
                if j < len(symbols) - 1:
                    pair = symbols[j] + symbols[j+1]
                    if pair in merges:
                        new_symbols.append(pair)
                        j += 2
                        made_merge = True
                        continue
                new_symbols.append(symbols[j])
                j += 1
            symbols = new_symbols
            if not made_merge:
                break
        return symbols
    
    def encode_sentence(self, sentence):
        norm = self.normalize(sentence)
        words = self.pretokenize(norm)
        token_ids = []
        for w in words:
            tokens = self.encode_word(w)
            token_ids.extend(self.token_to_id[t] for t in tokens if t in self.token_to_id)
        return token_ids
    
    def decode(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids]
        return "".join(tokens)


corpus = [
    "There is a big house",
    "I buy a house",
    "They buy a new house"
]
tokenizer = BPETokenizer(vocab_size=30)
tokenizer.train(corpus)

print(f"Merges: ")
for m in tokenizer.merges:
    print(f"\t{m}")

print(f"\nToken vocab:")
for t, i in tokenizer.token_to_id.items():
    print(f"\t{i}: {t}")

print(f"\nEncode: {tokenizer.encode_sentence('There is a big house')}\n")
