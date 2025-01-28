import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

def ciscenje_podataka():
    global data
    # Provjera nedostajućih podataka
    print(data.isnull().sum())

    # Zamjena nedostajućih vrijednosti
    data['tweet'] = data['tweet'].fillna('')
    data['likes'] = data['likes'].fillna(0)

    # Uklanjanje dupliciranih tweetova
    data = data.drop_duplicates(subset='tweet')

def vizualizacija_po_keywordu(keyword, save=False):
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
    plt.xticks(rotation=0)  # Oko X osi, okrenut na 0 stupnjeva za lakše čitanje
    
    for i, count in enumerate(keyword_popularity_counts):
        ax.text(i, count + 5, f'{count} ({popularity_percentages.iloc[i]:.1f}%)', ha='center', fontsize=12, color='black')

    if save:
        plt.savefig(f'visualisation/vizualizacija_{keyword}.png', bbox_inches='tight')
        print(f"Graf je spremljen kao 'visualisation/vizualizacija_{keyword}.png'")
    
    plt.show()

def usporedba_popularnosti_svih_kljucnih_rijeci(save=False):
    # Izračunaj postotke popularnosti za svaku ključnu riječ
    popularity_data = data.copy()
    popularity_data['popularity'] = popularity_data['likes'].apply(lambda x: 'high' if x > 500 else ('low' if x < 10 else 'middle'))
    
    keyword_popularity_counts = popularity_data.groupby(['keyword', 'popularity']).size().unstack(fill_value=0)
    
    keyword_popularity_percentage = keyword_popularity_counts.div(keyword_popularity_counts.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Kreiranje stacked bar plot-a
    keyword_popularity_percentage[['low', 'middle', 'high']].plot(kind='bar', stacked=True, color=['red', 'orange', 'green'], ax=ax)
    
    ax.set_title('Usporedba popularnosti tweetova po ključnim riječima', fontsize=14)
    ax.set_xlabel('Ključna riječ', fontsize=12)
    ax.set_ylabel('Postotak tweetova (%)', fontsize=12)
    ax.set_xticklabels(keyword_popularity_percentage.index, rotation=45, ha='right') 
    
    for i, keyword in enumerate(keyword_popularity_percentage.index):
        low_percentage = keyword_popularity_percentage.loc[keyword, 'low']
        middle_percentage = keyword_popularity_percentage.loc[keyword, 'middle']
        high_percentage = keyword_popularity_percentage.loc[keyword, 'high']
        
        # Prikaz postotka unutar segmenta
        ax.text(i, low_percentage / 2, f'{low_percentage:.1f}%', ha='center', fontsize=10, color='black')
        ax.text(i, low_percentage + middle_percentage / 2, f'{middle_percentage:.1f}%', ha='center', fontsize=10, color='black')
        ax.text(i, low_percentage + middle_percentage + high_percentage / 2, f'{high_percentage:.1f}%', ha='center', fontsize=10, color='black')
    
    if save:
        plt.savefig(f'visualisation/usporedba_popularnosti.png', bbox_inches='tight')
        print(f"Graf je spremljen kao 'visualisation/usporedba_popularnosti.png'")
    
    plt.tight_layout()
    plt.show()

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
def sentiment_pie_chart(keyword, save=False):
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

    if save:
        plt.savefig(f'visualisation/sentiment_{keyword}.png', bbox_inches='tight')
        print(f"Graf je spremljen kao 'visualisation/sentiment_{keyword}.png'")
    

    plt.show()


def treniranje_predikcija():
    data['popularity'] = data['likes'].apply(lambda x: 'high' if x > 500 else ('low' if x < 10 else 'middle'))

    # Kreiranje značajki iz tweetova (npr. broj riječi)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['tweet'])

    y = data['popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Kreiranje i treniranje modela
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Generiranje klasifikacijskog izvještaja
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv('classification_report.csv', index=True)
    print("Klasifikacijski izvještaj je spremljen u 'classification_report.csv'.")

    test_tweets = data.loc[y_test.index, 'tweet']
    predictions_df = pd.DataFrame({
        'Tweet': test_tweets,
        'Stvarna popularnost': y_test,
        'Predviđena popularnost': y_pred
    })

    # Spremanje predviđanja u CSV
    predictions_df.to_csv('predictions.csv', index=False)

    print("Predviđanja su spremljena u 'predictions.csv'.")

def report_visualisation():
    report_df = pd.read_csv('classification_report.csv')

    # Filtriramo samo metrike (precision, recall, f1-score) za klasu
    metrics = report_df.loc[:, ['precision', 'recall', 'f1-score']].iloc[:-3, :]  # Iznimka zadnja 3 reda (accuracy, macro avg, weighted avg)
    metrics.index = ['high', 'low', 'middle']

    # Plotanje bar chart-a za preciznost, odziv i f1-score
    metrics.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange', 'green'])
    plt.title('Klasifikacijski Izvještaj: Preciznost, Odziv i F1-score po Klasi', fontsize=14)
    plt.xlabel('Klasa', fontsize=12)
    plt.ylabel('Vrijednost', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'visualisation/report.png', bbox_inches='tight')
    plt.show()

data = pd.read_json('data/tweets.json', lines=True)
print(data.head())
print(f"Ukupan broj tweetova: {data.shape[0]}")


ciscenje_podataka()
print(data.head())
print(f"Ukupan broj tweetova: {data.shape[0]}")


# Pozivanje funkcije za keyword "COVID-19"
vizualizacija_po_keywordu("COVID-19", save=True)

# Pozivanje funkcije za sve tweetove
vizualizacija_po_keywordu("", save=True)

usporedba_popularnosti_svih_kljucnih_rijeci(save=True)

sentiment_pie_chart("Vaccine", save=True)

treniranje_predikcija()

report_visualisation()