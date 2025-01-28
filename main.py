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

            # Uklanjanje neutralnog sentimenta (ako postoji)
            if sentiment in ['POSITIVE', 'NEGATIVE']:
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

# Funkcija za vizualizaciju tweetova po popularnosti
def vizualizacija_po_keywordu(keyword):
    # Ako je keyword prazan, uzmi sve tweetove
    if keyword == "":
        keyword_data = data.copy()
        title = 'Broj tweetova po popularnosti (Svi tweetovi)'
    else:
        keyword_data = data[data['keyword'] == keyword].copy()
        title = f'Broj tweetova po popularnosti za keyword: {keyword}'

    keyword_data['popularity'] = keyword_data['likes'].apply(
        lambda x: 'high' if x > 500 else ('low' if x < 10 else 'middle')
    )

    keyword_popularity_counts = keyword_data['popularity'].value_counts()

    total_tweets = keyword_data.shape[0]

    popularity_percentages = (keyword_popularity_counts / total_tweets) * 100

    plt.figure(figsize=(8, 6))
    ax = keyword_popularity_counts.plot(kind='bar', color=['red', 'orange', 'green'])

    # Dodavanje naslova i oznaka
    plt.title(title, fontsize=14)
    plt.xlabel('Popularnost', fontsize=12)
    plt.ylabel('Broj tweetova', fontsize=12)

    # Dodavanje broja tweetova i postotaka iznad svake trake
    for i, count in enumerate(keyword_popularity_counts):
        ax.text(i, count + 5, f'{count} ({popularity_percentages.iloc[i]:.1f}%)', ha='center', fontsize=12, color='black')

    # Prikazivanje grafa
    plt.show()

# Funkcija za sentiment pie chart
def sentiment_pie_chart(keyword):
    # Učitaj rezultate analize sentimenta
    sentiment_df = pd.read_csv('sentiment_results.csv')

    # Filtriraj podatke prema ključnoj riječi
    keyword_data = sentiment_df[sentiment_df['keyword'] == keyword]

    # Ako nema tweetova za ključnu riječ, ispiši poruku
    if keyword_data.empty:
        print(f"Nema tweetova za keyword: {keyword}")
        return

    # Brojimo sentimenta
    sentiment_counts = keyword_data['sentiment'].value_counts()

    # Vizualizacija pie chart-a
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

# Funkcija za usporedbu popularnosti svih ključnih riječi
def usporedba_popularnosti_svih_kljucnih_rijeci():
    # Izračunaj postotke popularnosti za svaku ključnu riječ
    popularity_data = data.copy()
    popularity_data['popularity'] = popularity_data['likes'].apply(lambda x: 'high' if x > 500 else ('low' if x < 10 else 'middle'))

    # Grupiraj podatke po keyword-u i popularnosti
    keyword_popularity_counts = popularity_data.groupby(['keyword', 'popularity']).size().unstack(fill_value=0)

    # Izračunaj postotke za svaku ključnu riječ
    keyword_popularity_percentage = keyword_popularity_counts.div(keyword_popularity_counts.sum(axis=1), axis=0) * 100

    # Plotanje
    fig, ax = plt.subplots(figsize=(12, 8))

    # Kreiranje stacked bar plot-a
    keyword_popularity_percentage[['low', 'middle', 'high']].plot(kind='bar', stacked=True, color=['red', 'orange', 'green'], ax=ax)

    # Dodavanje naslova i oznaka
    ax.set_title('Usporedba popularnosti tweetova po ključnim riječima', fontsize=14)
    ax.set_xlabel('Ključna riječ', fontsize=12)
    ax.set_ylabel('Postotak tweetova (%)', fontsize=12)
    ax.set_xticklabels(keyword_popularity_percentage.index, rotation=45, ha='right')  # Okretanje oznaka za ključne riječi

    # Dodavanje postotka unutar svakog dijela trake
    for i, keyword in enumerate(keyword_popularity_percentage.index):
        low_percentage = keyword_popularity_percentage.loc[keyword, 'low']
        middle_percentage = keyword_popularity_percentage.loc[keyword, 'middle']
        high_percentage = keyword_popularity_percentage.loc[keyword, 'high']

        # Prikaz postotka unutar segmenta
        ax.text(i, low_percentage / 2, f'{low_percentage:.1f}%', ha='center', fontsize=10, color='white')
        ax.text(i, low_percentage + middle_percentage / 2, f'{middle_percentage:.1f}%', ha='center', fontsize=10, color='white')
        ax.text(i, low_percentage + middle_percentage + high_percentage / 2, f'{high_percentage:.1f}%', ha='center', fontsize=10, color='white')

    # Prikazivanje grafa
    plt.tight_layout()
    plt.show()

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

# Pozivanje funkcije za vizualizaciju tweetova prema ključnim riječima
vizualizacija_po_keywordu("COVID-19")

# Pozivanje funkcije za sve tweetove
vizualizacija_po_keywordu("")

# Pozivanje funkcije za usporedbu popularnosti ključnih riječi
usporedba_popularnosti_svih_kljucnih_rijeci()

# Pozivanje funkcije za pie chart sentiment analize
sentiment_pie_chart("Bitcoin")
