# SW Project

Αυτό τo repository είναι για την εργασία μας στο μάθημα Τεχνολόγια Λογισμικού 

Σφηναρολάκης Στέφανος inf2021218
Νικόλαος Τρυπάκης inf2021229
Ορέστης Ραφαήλ Μακρής inf2021129

Ιόνιο Πανεπιστήμιο Τμήμα Πληροφορικής

## How To Run App

Αρχικά, απαιτείται να κατέβει το Docker Desktop στο σύστημα.

Μετά, χρειάζεται δημιουργία του dockerfile:

```
FROM python:3.11.9
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/app.py"}
```

Μέσω αυτών των εντολών όπου αποτελούν το περιεχόμενο του
αρχείου "Dockerfile" ορίζεται τον ευρετήριο της εφαρμογής,
κατεβαίνουν οι απαραίτητες βιβλιοθήκες και μέσω του Streamlit
τρέχει το βασικό αρχείο της εφαρμογής.
 ́Επειτα την δημιουργία του αρχείου θα χρειαστεί να εκτελεστούν οι
ακόλουθες εντολές:
```
docker build -t "your_app_name" .
```
```
docker run -p 8501:8501 "your_app_name"
```
nai