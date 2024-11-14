# -*- coding: utf-8 -*-
"""Lab5_MNIST_EnsembleLearning_S.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WrPubpMJDD49P-n2liOHvMwuhOY_Ngyi

####  <span style='color:Blue'> Dane do analizy </span>:  MNIST.csv (Modified National Institute of Standards and Technology database)
####  <span style='color:Blue'> Cel badania </span>:  Rozpoznanie i klasyfikacji znaków ręcznie napisanych .

####   <span style='color:Blue'> Klasyfikator </span>:  Decision tree, RF, ExtraTree, XGBoost (Improved GradientBoosting) +PCA

##### <span style='color:Blue'>  Zadanie 1 </span>: Ładowanie zbioru treningowego, testowego, wstępna analiza  i dodawanie nazw dla kolumn
* Dodać nazwy kolumn: <span style='color:red'>  class, pixel1, pixel2,...pixel 784 </span>
* Wyznaczyć liczbę rekordów, liczbę cech w zbiorze treningowym i testowym.
* Wyznaczyć rozkład kategorii (w procentach).
* Narysować wykres słupkowy rozkładu kategorii.
##### <span style='color:Blue'>  Zadanie 2 </span>:   Wizualizacja 8 pierwszych liter w zbiorze uczącym i 8 pierwszych w zbiorze testowym.
__Wskazówka__: Przekształcić wektor 784-bitowy (wiersz) na macierz o wymiarach <span style='color:red'> $ 28 \times 28$ (reshape()) </span>.
##### <span style='color:Blue'>  Zadanie 3.1 </span>:  Tworzenie klasyfikatora <i>Decision tree</i>
__Wskazówka__: <span style='color:red'>  from sklearn.tree import DecisionTreeClassifier </span>

##### <span style='color:Blue'>  Zadanie 3.2 </span>:  Optymalizacja hyperparametrów DT,  ręczne ustawienie parametrów.
__Wskazówka: Zmienić parametry__
* Głębokość drzewa (max_depth)
* Kryterium podziału (criterion = 'gini', 'entropy')
* Miara oceny: accuracy
<b>Wniosek</b>: Które parametry są optymalne?
##### <span style='color:Blue'>  Zadanie 3.3 </span> : Wizualizacja drzewa. Analizując drzewo podać 5 najważniejszych  atrybutów.

* Głębokość drzewa (max_depth = 10, 20, 30)
* Kryterium podziału (criterion = 'gini', 'entropy', 'log_loss')
* Miara oceny: accuracy.
<b>Wniosek</b>: Które parametry są optymane?

##### <span style='color:Blue'>  Zadanie 4 </span> : Trenować zespół klasyfikatorów <i> RandomForest </i>  z optymalizacją parametrów.
__Wskazówka__: <span style='color:red'>  from sklearn.ensemble import RandomForestClassifier </span>

##### <span style='color:Blue'>  Zadanie 5 </span> : Trenować zespół klasyfikatorów <i> ExtraTree </i>  z optymalizacją parametrów.
__Wskazówka__: <span style='color:red'>  from sklearn.ensemble import ExtraTreeClassifier </span>

##### <span style='color:Blue'>  Zadanie 6 </span>:  Trenować zespół klasyfikatorów <i> XGBoost </i>  z optymalizacją parametrów.
__Wskazówka__:
<span style='color:red'>  !pip install  xgboost </span>
<span style='color:red'>  import xgboost as xgb  </span>

#####  <span style='color:Blue'>  Zadanie 7 </span>: Wykonać aumentację obrazów. Przesuwać każde zdjęcie (shift) do góry, w dół, w lewo i w prawo. Dla każdego zdjęcia tworzyć dodatkowe cztery obrazy i dodać je do zbioru treningowego.  

#####  <span style='color:Blue'>  Zadanie 8 </span>: Budować DT, RF, ExtraTree, XGBoost na transformowanych danych (aumentacją).

#####  <span style='color:Blue'>  Zadanie 9 </span>: Wyświetlić czas tworzenia modelu w zadaniu 8.
__Wskazówka__:

<span style='color:red'>   </span>
from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    
    image = image.reshape((28, 28))
    
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    
    return shifted_image.reshape([-1])

</span>

#####  <span style='color:Blue'>  Zadanie 10 </span>: Napisać podsumowanie:
1. Dla danych oryginalnych: Który  
klasyfikator jest najlepszy i z jakimi parametrami?
2. Dla danych z augmentacją: Który klasyfikator jest najlepszy?
3. Czy warto zrobić aumentację obrazów?

## Zadanie 1: Ładowanie i wstępna analiza
"""

#Importowanie danych z lokalnego dysku
import pandas as pd
column_names = [f'pixel{i+1}' for i in range(784)]
test_set = pd.read_csv('mnist_test.csv')
train_set = pd.read_csv('mnist_train.csv')

#data_frame.info()
df_test = pd.DataFrame(test_set, columns = column_names)
df_train = pd.DataFrame(train_set, columns = column_names)
print(f"Liczba rekordów {test_set.shape[0]} i cech {test_set.shape[1]}.")
print(f"Liczba rekordów {train_set.shape[0]} i cech {train_set.shape[1]}.")

# Rozkład kategorii.
test_distribution = test_set.value_counts()
train_distribution = train_set.value_counts()

# Rozkładu kategorii, wykres słupkowy
import matplotlib.pyplot as plt

test_distribution.plot(kind='bar') #bar = wykres słupkowy
plt.xlabel('Category')
plt.ylabel('Pixel')
plt.title('Distribution')
plt.show()
train_distribution.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Pixel')
plt.title('Distribution')
plt.show()

# Podział danych:  zbiór treningowy i testowy
#from sklearn.model_selection import train_test_split

#train,test = train_test_split (df, test_size=0.3, random_state=50, shuffle = True)

# Tworzyć atrybut docelowy.

X_train = df_train.iloc[:,1:785]
y_train = df_train.iloc[:,0]
X_test = df_test.iloc[:,1:785]
y_test = df_test.iloc[:,0]

print(X_train.shape)
print(X_test.shape)

"""##  Zadanie 2: Wyświetlenie 8 pierwszych obrazów"""

import matplotlib.pyplot as plt
import numpy as np
for i in range(8):
     # define subplot
    plt.subplot(240+1+i)
    # plot raw pixel data
    ith_image = X_train.iloc[i,:]
    ith_image_arr = ith_image.to_numpy()
    ith_image= ith_image_arr.reshape(28,28)
    plt.imshow(ith_image, cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

"""## Zadanie 3: Budowanie klasyfikatora DT"""

# Klasyfikator DT
# Ewaluacja modelu: accuracy-score,

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score

from sklearn import tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
f_names = list(df_train.columns.values.tolist())
f_names = f_names[1:]
t_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
tree.plot_tree(tree_clf, max_depth =1, feature_names = f_names,
                   class_names=t_names, filled = True)
plt.show()

# DT - Text form
from sklearn.tree import export_text
f_names = list(df_train.columns.values.tolist())
f_names = f_names[1:]
r = export_text(tree_clf, feature_names = f_names, max_depth = 3)
print(r)

# RF classifier

# ExtraTree
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

# Accuracy ExtraTree

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(n_estimators = 20, criterion ='gini')
xgb_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)
predictions = xgb_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, predictions))
#print(confusion_matrix(y_test, predictions))
#print('F1 measure:', f1_score (y_test, predictions))
print(classification_report(y_test, predictions))

from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

image = X_train.iloc[1000,:]
image_arr = image.to_numpy()
image = image_arr.reshape((28,28))
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]