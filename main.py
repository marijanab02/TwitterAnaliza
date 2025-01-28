import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np 

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

def ciscenje_podataka():
    global data
    # Provjera nedostajućih podataka
    print(data.isnull().sum())

    # Zamjena nedostajućih vrijednosti
    data['tweet'] = data['tweet'].fillna('')
    data['likes'] = data['likes'].fillna(0)

    # Uklanjanje dupliciranih tweetova
    data = data.drop_duplicates(subset='tweet')

def sentiment():
    classifier = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')

    # Predikcija sentimenta
    result = classifier("I love learning AI!")
    print(result)

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

ciscenje_podataka()
print(data.head())
print(f"Ukupan broj tweetova: {data.shape[0]}")


sentiment()


# Pozivanje funkcije za keyword "COVID-19"
vizualizacija_po_keywordu("COVID-19")

# Pozivanje funkcije za sve tweetove
vizualizacija_po_keywordu("")


usporedba_popularnosti_svih_kljucnih_rijeci()
