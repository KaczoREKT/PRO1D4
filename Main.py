import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    # Zad 1
    print("Zad 1")
    test_set = pd.read_csv('mnist_test.csv')
    train_set = pd.read_csv('mnist_train.csv')
    print(f"Liczba rekordów {test_set.shape[0]} i cech {test_set.shape[1]}.")
    print(f"Liczba rekordów {train_set.shape[0]} i cech {train_set.shape[1]}.")