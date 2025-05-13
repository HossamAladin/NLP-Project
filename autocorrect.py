import torch
import re
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)

def preprocess(sentence: str) -> str:
    """Enhanced Arabic text preprocessing"""
    sentence = sentence.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    arabic_chars = (
        'ابتثجحخدذرزسشصضطظعغفقكلمنهويءئؤىة'
        'ًٌٍَُِّْٰ'
        '0123456789'
        '.,!؟؛،'
    )
    return ''.join([c for c in sentence if c in arabic_chars or c.isspace()])

def data_vocab(dataframe, min_freq=3):
    """Create vocabulary with frequency filtering"""
    words_freq = Counter()
    for text in dataframe['text']:
        words_freq.update(text.split())
    return {word: freq for word, freq in words_freq.items() if freq >= min_freq}

def find_misspellings(text: str, vocab: dict, threshold: float = 0.15) -> list:
    """Identify potentially misspelled words using MLM probability and vocab"""
    words = text.split()
    misspelled_indices = []

    for i, word in enumerate(words):
        if word not in vocab:
            masked_words = words.copy()
            masked_words[i] = tokenizer.mask_token
            masked_sentence = " ".join(masked_words)

            inputs = tokenizer(masked_sentence, return_tensors="pt").to(model.device)
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, mask_token_index]
                probs = torch.softmax(logits, dim=-1).squeeze()
                word_id = tokenizer.encode(word, add_special_tokens=False)
                word_prob = torch.mean(probs[word_id]) if word_id else 0

            if word_prob < threshold:
                misspelled_indices.append(i)

    return misspelled_indices

def generate_masked_sentences(text: str, misspelled_indices: list) -> list:
    """Generate masked sentences for each misspelled word"""
    words = text.split()
    return [
        " ".join(words[:idx] + [tokenizer.mask_token] + words[idx + 1:])
        for idx in misspelled_indices
    ]

def predict(masked_sentence: str, top_k=5) -> str:
    """Predict masked word in sentence using MLM"""
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_token_index]
    probs = torch.softmax(logits, dim=-1).squeeze()
    top_k_tokens = torch.topk(probs, top_k)

    for token_id in top_k_tokens.indices:
        token = tokenizer.decode([token_id]).strip()
        if re.match(r'^[\u0600-\u06FF]{2,}$', token):
            return token

    return "[UNK]"

def pipeline(input_text: str, verbose: bool = True) -> str:
    """
    Full pipeline for Arabic spelling correction.
    """
    processed_text = preprocess(input_text)
    
    # Load vocabulary (you may want to save/load this from a file for production)
    try:
        import pickle
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except:
        print("Warning: No vocabulary file found. Using empty vocabulary.")
        vocab = {}
    
    misspelled_indices = find_misspellings(processed_text, vocab)

    if not misspelled_indices:
        if verbose:
            print("✅ لا توجد أخطاء إملائية واضحة.")
        return processed_text

    masked_sentences = generate_masked_sentences(processed_text, misspelled_indices)
    words = processed_text.split()
    corrections = {}

    for idx, masked in zip(misspelled_indices, masked_sentences):
        correction = predict(masked)
        corrections[words[idx]] = correction
        words[idx] = correction

    corrected_sentence = " ".join(words)

    if verbose:
        print("🔍 الكلمات التي تم تصحيحها:")
        for original, corrected in corrections.items():
            print(f" - {original} ➤ {corrected}")

    return corrected_sentence

if __name__ == "__main__":
    # Example usage
    input_text = input("Enter Arabic text to correct: ")
    corrected = pipeline(input_text)
    print("Corrected text:", corrected)