from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Набор простых текстов
documents = [
    "The economy is growing fast this year.",
    "Stock markets reached new highs.",
    "The inflation rate is under control.",
    "New tech startups are booming in Silicon Valley.",
    "Artificial Intelligence is the future of technology.",
    "Machine learning powers modern applications.",
    "The new movie was an amazing thriller.",
    "The film had great acting and direction.",
    "I love watching science fiction films.",
]

# Векторизация текста (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Применяем k-means
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Вывод результатов
labels = kmeans.labels_

for i, doc in enumerate(documents):
    print(f"Cluster {labels[i]}: {doc}")
