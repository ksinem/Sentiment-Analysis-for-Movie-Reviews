import pandas as pd
import re
from ydata_profiling import ProfileReport
import nltk
from nltk.corpus import stopwords


def clean_and_remove_stopwords(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)


data = pd.read_csv("IMDB Dataset.csv")
data["reviews_clean"] = data["review"].apply(clean_and_remove_stopwords)
# Generate col containing the length of each review
data['review_length'] = data['reviews_clean'].apply(lambda x: len(x.split()))
profile = ProfileReport(data, title="IMDB Review Data", explorative=True)
profile.to_file("imdb_report.html")


