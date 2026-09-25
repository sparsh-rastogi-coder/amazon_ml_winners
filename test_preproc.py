import re
import unidecode
import wordninja

NOISE_WORDS_REGEX = re.compile(
    r'\b(llc|inc|ltd|pvt|private|limited|corp|corporation|services|center|partners|enterprises|co|m/?s|mr|mrs|smt)\b', 
    re.IGNORECASE
)

def preprocess_text(text, is_name=True):
    if not text: return ""
    text = unidecode.unidecode(str(text)).lower()
    
    if is_name:
        tokens = []
        for word in text.split():
            if re.search(r'\.(com|in|net|org|co\.in|us|info)$', word):
                word = re.sub(r'\.(com|in|net|org|co\.in|us|info)$', '', word)
                tokens.extend(wordninja.split(word))
            else:
                tokens.append(word)
        text = " ".join(tokens)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    if is_name:
        text = NOISE_WORDS_REGEX.sub(' ', text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

tests = [
    ("tmrindia.com", "tmr india"),
    ("Future Healthcare Pvt Ltd", "future healthcare"),
    ("फ्यूचर हेल्थकेयर प्रा. लि.", "phyuucr helthkeyr praa li"),
    ("B+ Rétail Incorporated", "b retail incorporated"),
    ("@rockytrust", "rockytrust"),
    ("kingaria.com", "king aria")
]

for orig, exp in tests:
    res = preprocess_text(orig)
    print(f"'{orig}' -> '{res}'")
