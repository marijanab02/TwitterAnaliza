# Uloga umjetne inteligencije u analizi podataka o društvenim mrežama

Ovaj projekt analizira i predviđa popularnost tweetova koristeći obradu prirodnog jezika i strogo učenje. Projekt uključuje čišćenje podataka, vizualizaciju, analizu sentimenta i klasifikaciju popularnosti tweetova na temelju broja lajkova.



## Postavljanje projekta

### Instalacija potrebnih paketa

```bash
pip install -r requirements.txt
```

### Pokretanje analize
```bash
python main.py
```



## Čišćenje podataka
Skripta main.py poziva funkciju ciscenje_podataka() koja zamjenjuje nedostajuće vrijednosti i uklanja duplicirane tweetove



## Vizualizacija podataka

**Vizualizacija popularnosti tweetova za određeni keyword:**
   ```python
   vizualizacija_po_keywordu("COVID-19", save=True)
   ```
Graf je spremljen u visualisation/vizualizacija_COVID-19.png

**Vizualizacija svih tweetova:**
   ```python
   vizualizacija_po_keywordu("", save=True)
   ```
Graf je spremljen u visualisation/vizualizacija_svi.png

**Usporedba popularnosti svih ključnih riječi:**
   ```python
   usporedba_popularnosti_svih_kljucnih_rijeci(save=True)
   ```
Spremljeno u visualisation/usporedba_popularnosti.png


## Analiza sentimenta

**Pokretanje analize sentimenta:**
   ```python
   analiziraj_sentiment(data, num_tweets=10)
   ```
Rezultati su spremljeni u sentiment_results.csv

**Vizualizacija sentimenta za ključnu riječ (npr. "Vaccine")**
   ```python
   sentiment_pie_chart("Vaccine", save=True)
   ```
Spremljeno u visualisation/sentiment_Vaccine.png



## Treniranje modela

**Treniranje i predviđanje popularnosti tweetova**
   ```python
   treniranje_predikcija()
   ```
Generira klasifikacijski izvještaj classification_report.csv i predviđanja predictions.csv
Izvještaj prikazuje preciznost, odziv i F1-score za tri klase popularnosti tweetova:
- **high** (više od 500 lajkova)
- **middle** (između 10 i 500 lajkova)
- **low** (manje od 10 lajkova)

**Vizualizacija klasifikacijskog izvještaja**
   ```python
   report_visualisation()
   ```
Spremljeno u visualisation/report.png




## Autori
- [Ivana Stojić](https://github.com/ivanastojic)
- [Marijana Bandić](https://github.com/marijanab02)
