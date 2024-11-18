import pandas as pd
import matplotlib.pyplot as plt

#=======================================================================================================================
#                                                      ZADANIE 1
#=======================================================================================================================
# Ładowanie zbioru danych (przykład: zbiór MNIST)
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Dodanie nazw kolumn
columns = ['class'] + [f'pixel{i+1}' for i in range(784)]
train_data.columns = columns
test_data.columns = columns

# Wyświetlenie informacji o zbiorze treningowym
print(f"Liczba rekordów w zbiorze treningowym: {train_data.shape[0]}")
print(f"Liczba cech w zbiorze treningowym: {train_data.shape[1]}")

# Wyznaczenie rozkładu kategorii
category_distribution = train_data['class'].value_counts(normalize=True) * 100
print(f"Rozkład kategorii (w procentach):\n{category_distribution}")

# Wykres słupkowy rozkładu kategorii
category_distribution.plot(kind='bar')
plt.title('Rozkład kategorii')
plt.xlabel('Klasa')
plt.ylabel('Procent')
plt.show()

#=======================================================================================================================
#                                                      ZADANIE 2
#=======================================================================================================================
import numpy as np

# Funkcja do wyświetlania obrazu
def show_image(data, index):
    image = data.iloc[index, 1:].values.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Klasa: {data.iloc[index, 0]}")
    plt.show()

# Wizualizacja pierwszych 8 obrazów z zestawu uczącego
for i in range(8):
    show_image(train_data, i)

# Wizualizacja pierwszych 8 obrazów z zestawu testowego
for i in range(8):
    show_image(test_data, i)
#=======================================================================================================================
#                                                      ZADANIE 3.1
#=======================================================================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Przygotowanie danych
X_train = train_data.iloc[:, 1:]
y_train = train_data['class']
X_test = test_data.iloc[:, 1:]
y_test = test_data['class']

# Tworzenie i trenowanie klasyfikatora
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Ocena modelu
accuracy = dt.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
#=======================================================================================================================
#                                                      ZADANIE 3.2
#=======================================================================================================================
# Ręczna zmiana parametrów
dt = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

# Ocena modelu
accuracy = dt.score(X_test, y_test)
print(f"Accuracy (max_depth=10, criterion='entropy'): {accuracy}")

# Zmiana parametrów
dt = DecisionTreeClassifier(max_depth=20, criterion='gini', random_state=42)
dt.fit(X_train, y_train)

# Ocena modelu
accuracy = dt.score(X_test, y_test)
print(f"Accuracy (max_depth=20, criterion='gini'): {accuracy}")


#=======================================================================================================================
#                                                      ZADANIE 3.3
#=======================================================================================================================
from sklearn.tree import plot_tree

# Wizualizacja drzewa
plt.figure(figsize=(20,10))
plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=[str(i) for i in y_train.unique()])
plt.show()

# Analiza najważniejszych atrybutów
importance = dt.feature_importances_
important_features = np.argsort(importance)[-5:]  # 5 najważniejszych cech
print(f"5 najważniejszych cech: {X_train.columns[important_features]}")

#=======================================================================================================================
#                                                      ZADANIE 4
#=======================================================================================================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
accuracy_rf = rf.score(X_test, y_test)
print(f"Accuracy (RandomForest): {accuracy_rf}")

#=======================================================================================================================
#                                                      ZADANIE 5
#=======================================================================================================================
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(random_state=42)
et.fit(X_train, y_train)
accuracy_et = et.score(X_test, y_test)
print(f"Accuracy (ExtraTree): {accuracy_et}")

#=======================================================================================================================
#                                                      ZADANIE 6
#=======================================================================================================================
import xgboost as xgb

# Przygotowanie danych dla XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train)),
    'eval_metric': 'merror'
}

# Trenowanie modelu
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predykcja i ocena
y_pred = bst.predict(dtest)
accuracy_xgb = np.mean(y_pred == y_test)
print(f"Accuracy (XGBoost): {accuracy_xgb}")

#=======================================================================================================================
#                                                      ZADANIE 7
#=======================================================================================================================
from scipy.ndimage import shift

def augment_images(data):
    augmented_images = []
    for i in range(data.shape[0]):
        image = data.iloc[i, 1:].values.reshape(28, 28)
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # przesunięcia: góra, dół, lewo, prawo
            shifted_image = shift(image, shift=direction, mode='nearest')
            augmented_images.append(shifted_image.flatten())
    return np.array(augmented_images)

augmented_data = augment_images(train_data)
augmented_labels = np.tile(y_train, 5)  # Dublujemy etykiety
#=======================================================================================================================
#                                                      ZADANIE 8
#=======================================================================================================================
# Ponowne trenowanie modeli na danych z augmentacją
rf_augmented = RandomForestClassifier(random_state=42)
rf_augmented.fit(augmented_data, augmented_labels)

et_augmented = ExtraTreesClassifier(random_state=42)
et_augmented.fit(augmented_data, augmented_labels)

xgb_augmented = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(augmented_labels)), eval_metric='merror')
xgb_augmented.fit(augmented_data, augmented_labels)

# Oceny
print(f"Accuracy (RandomForest Augmented): {rf_augmented.score(X_test, y_test)}")
print(f"Accuracy (ExtraTree Augmented): {et_augmented.score(X_test, y_test)}")
print(f"Accuracy (XGBoost Augmented): {xgb_augmented.score(X_test, y_test)}")
#=======================================================================================================================
#                                                      ZADANIE 9
#=======================================================================================================================
import time

start_time = time.time()
rf_augmented.fit(augmented_data, augmented_labels)
end_time = time.time()
print(f"Czas trenowania RandomForest: {end_time - start_time} sekundy")

#=======================================================================================================================
#                                                      ZADANIE 10
#=======================================================================================================================
result = "rezultat: "
print(result)
