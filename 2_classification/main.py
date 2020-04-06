import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


if __name__ == '__main__':
    # Чтение файлов
    df = pd.read_csv('csvs/zoo.csv')
    print('Животные: ')
    print(df)

    # Все параметры животного
    features = list(df.columns)
    print()
    print('Все параметры животного: ')
    print(features)

    # Оставляем только параметры для классификации
    features.remove('class_type')
    features.remove('animal_name')
    print()
    print('Параметры животного для классификации: ')
    print(features)

    # Составляем матрицы для классификации и проверки
    X = df[features].values.astype(np.float32)
    Y = df.class_type
    # Размеры матриц X (животные) и Y (номер класса для проверки)
    print()
    print('Размеры матриц X (животные) и Y (номер класса для проверки)')
    print(X.shape)
    print(Y.shape)

    # Составление матриц для тренировки и тестирования
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    print()
    print('Размеры матриц X и Y для тренировки: ')
    print(X_train.shape)
    print(Y_train.shape)
    print('Размеры матриц X и Y для тестов: ')
    print(X_test.shape)
    print(Y_test.shape)

    # Логистическая регрессия
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print()
    print('Логистическая регрессия: ')
    print("training accuracy :", model.score(X_train, Y_train))
    print("testing accuracy :", model.score(X_test, Y_test))

    # Дерево решений
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    print()
    print('Дерево решений: ')
    print("training accuracy :", model.score(X_train, Y_train))
    print("testing accuracy :", model.score(X_test, Y_test))

    # Random forest
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    print()
    print('Random forest: ')
    print("training accuracy :", model.score(X_train, Y_train))
    print("testing accuracy :", model.score(X_test, Y_test))

    # Ada boost
    model = AdaBoostClassifier()
    model.fit(X_train, Y_train)
    print()
    print('Ada boost: ')
    print("training accuracy :", model.score(X_train, Y_train))
    print("testing accuracy :", model.score(X_test, Y_test))
