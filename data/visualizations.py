import pandas as pd
import re
from ydata_profiling import ProfileReport

data = pd.read_csv("IMDB Dataset.csv")

data["reviews_clean"] = data["review"].apply(lambda word: re.sub(r'<[^>]+>', '', str(word)))

data['review_length'] = data['reviews_clean'].apply(lambda x: len(x.split()))
print(data.describe())
profile = ProfileReport(data, title="IMDB Review Data")
profile.to_file("imdb_report.html")



