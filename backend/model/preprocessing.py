import nltk
nltk.download('stopwords')
import pandas as pd
import re
import string
from nltk.corpus import stopwords

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data[["text", "label"]]

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

data.to_csv("dataset/cleaned_data.csv", index=False)
