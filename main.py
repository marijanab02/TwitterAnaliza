import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# Funkcija za analizu sentimenta
def analiziraj_sentiment(data, num_tweets=10):
    # Kreiraj sentiment analizator
    classifier = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')

    # Lista za pohranu rezultata analize sentimenta
    sentiment_results = []

    # Odaberi prvih 10 tweetova za svaku ključnu riječ
    keywords = data['keyword'].unique()[:5]  # Uzmi prvih 5 ključnih riječi
    for keyword in keywords:
        # Filtriraj tweetove prema ključnoj riječi
        keyword_data = data[data['keyword'] == keyword].head(num_tweets)  # Uzimamo samo prvih 10 tweetova

        for _, row in keyword_data.iterrows():
            tweet = row['tweet']
            # Predikcija sentimenta
            sentiment_result = classifier(tweet)[0]
            sentiment = sentiment_result['label']  # 'LABEL_1' (pozitivno) ili 'LABEL_0' (negativno)
            confidence_score = sentiment_result['score']  # Sigurnosni score

            # Spremi rezultate u listu
            sentiment_results.append({
                'keyword': keyword,
                'tweet': tweet,
                'sentiment': sentiment,
                'confidence_score': confidence_score
            })

    # Spremi rezultate u DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)

    # Spremi u CSV
    sentiment_df.to_csv('sentiment_results.csv', index=False)
    print("Analiza sentimenta završena. Rezultati su spremljeni u 'sentiment_results.csv'.")

# Funkcija za vizualizaciju sentimenta kao Pie Chart (bez neutralnih tweetova)
def sentiment_pie_chart(keyword):
    # Učitaj rezultate analize sentimenta
    sentiment_df = pd.read_csv('sentiment_results.csv')
    
    # Filtriraj podatke prema ključnoj riječi
    keyword_data = sentiment_df[sentiment_df['keyword'] == keyword]

    # Ako nema tweetova za ključnu riječ, ispiši poruku
    if keyword_data.empty:
        print(f"Nema tweetova za keyword: {keyword}")
        return

    # Filtriraj samo pozitivne i negativne tweetove
    keyword_data = keyword_data[keyword_data['sentiment'].isin(['POSITIVE', 'NEGATIVE'])]

    # Brojimo sentimenta
    sentiment_counts = keyword_data['sentiment'].value_counts()

    # Ako nedostaje bilo koji od sentimenta, dodaj ga s vrijednošću 0
    if 'POSITIVE' not in sentiment_counts:
        sentiment_counts['POSITIVE'] = 0
    if 'NEGATIVE' not in sentiment_counts:
        sentiment_counts['NEGATIVE'] = 0

    # Vizualizacija pie chart-a s novim bojama (pozitivno zeleno, negativno crveno)
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    plt.title(f'Sentiment analiza za keyword: {keyword}')
    plt.ylabel('')  # Ukloniti labelu jer nije potrebna
    plt.show()

# Funkcija za čišćenje podataka
def ciscenje_podataka():
    global data
    # Provjera nedostajućih podataka
    print(data.isnull().sum())

    # Zamjena nedostajućih vrijednosti
    data['tweet'] = data['tweet'].fillna('')
    data['likes'] = data['likes'].fillna(0)

    # Uklanjanje dupliciranih tweetova
    data = data.drop_duplicates(subset='tweet')

# Učitavanje podataka iz JSON datoteke
data = pd.read_json('data/tweets.json', lines=True)

# Pregled prvih nekoliko redova
print(data.head())
print(f"Ukupan broj tweetova: {data.shape[0]}")

# Čišćenje podataka
ciscenje_podataka()
print(data.head())
print(f"Ukupan broj tweetova: {data.shape[0]}")

# Pozivanje funkcije za analizu sentimenta
analiziraj_sentiment(data)

# Pozivanje funkcije za pie chart sentimenta za keyword " "
sentiment_pie_chart("Vaccine")
