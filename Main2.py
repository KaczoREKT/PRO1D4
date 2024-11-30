# Importowanie bibliotek
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Zad 1

# Ładowanie danych
train_data = pd.read_csv("mnist_train.csv", header=None)
test_data = pd.read_csv("mnist_test.csv", header=None)

# Dodawanie nazw kolumn
column_names = ['class'] + [f'pixel{i}' for i in range(1, 785)]
train_data.columns = column_names
test_data.columns = column_names

# Wstępna analiza
print(f"Rozmiar zbioru treningowego: {train_data.shape}")
print(f"Rozmiar zbioru testowego: {test_data.shape}")

# Wyznaczanie rozkładu kategorii
train_distribution = train_data['class'].value_counts(normalize=True) * 100
print("\nRozkład kategorii w zbiorze treningowym (w procentach):")
print(train_distribution)

# Wykres słupkowy rozkładu kategorii
plt.figure(figsize=(10, 6))
plt.bar(train_distribution.index, train_distribution.values, color='skyblue', edgecolor='black')
plt.title("Rozkład kategorii w zbiorze treningowym")
plt.xlabel("Klasa")
plt.ylabel("Procentowy udział")
plt.xticks(range(10))  # Klasy od 0 do 9
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Zad 2
from sklearn.decomposition import PCA

# Separacja danych i etykiet
X_train = train_data.drop(columns=['class']).values
y_train = train_data['class'].values
X_test = test_data.drop(columns=['class']).values
y_test = test_data['class'].values

# Standaryzacja danych
X_train = X_train / 255.0
X_test = X_test / 255.0

# PCA dla 90% wyjaśnionej wariancji
pca = PCA(n_components=0.90)
pca.fit(X_train)

# Liczba wymiarów
n_dimensions_90 = pca.n_components_
print(f"Liczba wymiarów dla 90% wyjaśnionej wariancji: {n_dimensions_90}")

# Zad 2.2

explained_variance_ratios = []
dimensions = range(10, 151)

for n in dimensions:
    pca = PCA(n_components=n)
    pca.fit(X_train)
    explained_variance_ratios.append(sum(pca.explained_variance_ratio_))

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(dimensions, explained_variance_ratios, marker='o', color='blue')
plt.axhline(y=0.90, color='red', linestyle='--', label='90% wyjaśnionej wariancji')
plt.title("Wyjaśniona wariancja a liczba wymiarów")
plt.xlabel("Liczba wymiarów")
plt.ylabel("Stopień wyjaśnionej wariancji")
plt.legend()
plt.grid()
plt.show()

# Zad 3: Redukcja wymiarów i transformacja danych
print("Zad 3")
# Redukcja wymiarów PCA dla 90% wyjaśnionej wariancji
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Informacje o nowym zbiorze danych
print(f"Rozmiar danych po PCA (zbiór treningowy): {X_train_pca.shape}")
print(f"Rozmiar danych po PCA (zbiór testowy): {X_test_pca.shape}")

# Zad 4: Wizualizacja danych oryginalnych i skompresowanych
print("Zadanie 4: Wizualizacja danych oryginalnych i skompresowanych")

# Redukcja danych do 2 wymiarów dla wizualizacji
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train)

# Tworzenie wykresu dla danych oryginalnych
plt.figure(figsize=(12, 6))

# Oryginalne dane w PCA 2D
plt.subplot(1, 2, 1)
plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=y_train, cmap='tab10', s=1)
plt.title("Dane oryginalne po redukcji do 2 wymiarów (PCA)")
plt.xlabel("Składowa 1")
plt.ylabel("Składowa 2")
plt.colorbar(label="Klasa")
plt.grid()

# Dane skompresowane (po PCA z 90% wyjaśnionej wariancji)
plt.subplot(1, 2, 2)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab10', s=1)
plt.title("Dane skompresowane po PCA (2 pierwsze wymiary)")
plt.xlabel("Składowa 1")
plt.ylabel("Składowa 2")
plt.colorbar(label="Klasa")
plt.grid()

plt.tight_layout()
plt.show()
# Zad 5: Trenowanie klasyfikatora DT na danych oryginalnych
print("Zadanie 5: Trenowanie klasyfikatora DT na danych oryginalnych")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Definiowanie parametrów do optymalizacji
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Inicjalizacja klasyfikatora i optymalizacja parametrów
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Pomiar czasu trenowania
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Najlepszy model i czas trenowania
best_dt = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Dokładność na zbiorze walidacyjnym: {grid_search.best_score_:.4f}")
print(f"Czas trenowania: {end_time - start_time:.2f} sekundy")

# Zad 6: Trenowanie klasyfikatora DT na danych skompresowanych
print("Zadanie 6: Trenowanie klasyfikatora DT na danych skompresowanych")

# Pomiar czasu trenowania
start_time = time.time()
grid_search.fit(X_train_pca, y_train)
end_time = time.time()

# Najlepszy model i czas trenowania
best_dt_pca = grid_search.best_estimator_
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Dokładność na zbiorze walidacyjnym: {grid_search.best_score_:.4f}")
print(f"Czas trenowania na danych skompresowanych: {end_time - start_time:.2f} sekundy")

# Zad 7: Trenowanie klasyfikatora Logistic Regression na danych oryginalnych
print("Zadanie 7: Trenowanie klasyfikatora Logistic Regression na danych oryginalnych")

from sklearn.linear_model import LogisticRegression

# Definiowanie parametrów do optymalizacji
param_grid_lr = {
    'penalty': ['l2', 'l1', 'elasticnet', None],  # Poprawienie wartości 'none' na None
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear', 'saga']  # Dodanie solvera 'saga', który obsługuje 'elasticnet'
}

# Inicjalizacja klasyfikatora i optymalizacja parametrów
lr = LogisticRegression(max_iter=1000, random_state=42)
grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=3, scoring='accuracy', n_jobs=-1)

# Pomiar czasu trenowania
start_time = time.time()
grid_search_lr.fit(X_train, y_train)
end_time = time.time()

# Najlepszy model i czas trenowania
best_lr = grid_search_lr.best_estimator_
print(f"Najlepsze parametry: {grid_search_lr.best_params_}")
print(f"Dokładność na zbiorze walidacyjnym: {grid_search_lr.best_score_:.4f}")
print(f"Czas trenowania: {end_time - start_time:.2f} sekundy")
# Zad 8: Trenowanie klasyfikatora k-NN na danych oryginalnych i zredukowanych
print("Zadanie 8: Trenowanie klasyfikatora k-NN na danych oryginalnych i zredukowanych")

from sklearn.neighbors import KNeighborsClassifier

# Parametry do optymalizacji dla k-NN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Inicjalizacja klasyfikatora k-NN
knn = KNeighborsClassifier()

# Optymalizacja parametrów dla danych oryginalnych
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='accuracy', n_jobs=-1)

# Pomiar czasu trenowania na danych oryginalnych
start_time = time.time()
grid_search_knn.fit(X_train, y_train)
end_time = time.time()
best_knn = grid_search_knn.best_estimator_
print(f"Najlepsze parametry (oryginalne): {grid_search_knn.best_params_}")
print(f"Dokładność na zbiorze walidacyjnym (oryginalne): {grid_search_knn.best_score_:.4f}")
print(f"Czas trenowania na danych oryginalnych: {end_time - start_time:.2f} sekundy")

# Pomiar czasu trenowania na danych skompresowanych
start_time = time.time()
grid_search_knn.fit(X_train_pca, y_train)
end_time = time.time()
best_knn_pca = grid_search_knn.best_estimator_
print(f"Najlepsze parametry (skompresowane): {grid_search_knn.best_params_}")
print(f"Dokładność na zbiorze walidacyjnym (skompresowane): {grid_search_knn.best_score_:.4f}")
print(f"Czas trenowania na danych skompresowanych: {end_time - start_time:.2f} sekundy")
# Zad 9: Wizualizacja danych na dwóch wymiarach wyznaczonych przez t-SNE i PCA
print("Zadanie 9: Wizualizacja danych na dwóch wymiarach wyznaczonych przez t-SNE i PCA")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)

# PCA
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train)

# Tworzenie wykresów
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Wizualizacja t-SNE
ax1.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap='tab10', s=1)
ax1.set_title('Wizualizacja danych za pomocą t-SNE')
ax1.set_xlabel('Składowa 1')
ax1.set_ylabel('Składowa 2')

# Wizualizacja PCA
ax2.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=y_train, cmap='tab10', s=1)
ax2.set_title('Wizualizacja danych za pomocą PCA')
ax2.set_xlabel('Składowa 1')
ax2.set_ylabel('Składowa 2')

# Pokaż wykres
plt.tight_layout()
plt.show()

