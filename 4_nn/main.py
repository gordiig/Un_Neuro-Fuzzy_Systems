import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense


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
    Y = df.class_type.values

    # Решейпим список лейблов в матрицу для Y
    encoder = OneHotEncoder()  # using encoding of class_type as this is a multi class problem.
    Y = encoder.fit_transform(Y.reshape(-1, 1)).toarray()  # fitting our data to encoder.

    # Размеры матриц X (животные) и Y (номер класса для проверки)
    print()
    print('Размеры матриц X (животные) и Y (номер класса для проверки)')
    print(X.shape)
    print(Y.shape)

    # Составление матриц для тренировки и тестирования
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print()
    print('Размеры матриц X и Y для тренировки: ')
    print(X_train.shape)
    print(Y_train.shape)
    print('Размеры матриц X и Y для тестов: ')
    print(X_test.shape)
    print(Y_test.shape)

    # Модель
    model = Sequential()

    # Добавляем слои
    model.add(Dense(units=20, activation='relu', input_dim=16))  # hiddenlayer1
    model.add(Dense(units=10, activation='relu'))  # hiddenlayer2
    model.add(Dense(units=7, activation='softmax'))  # outputlayer

    # Компиляция модели
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Тренировка сети и малидация с помощью тестовых данных
    model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_data=(X_test, Y_test))

    # Процент правильных угадываний
    print(f'Точность модели = {model.evaluate(X_test, Y_test)[1]}')

    # Прогоняем нейронку для каждой животины из тестовой выборки, и выводим результаты
    y_pred_con = model.predict(X_test)
    y_pred, y_correct = [], []

    for i in Y_test:
        y_correct.append(np.argmax(i))
    for j in y_pred_con:
        y_pred.append(np.argmax(j))

    pred_df = pd.DataFrame()
    pred_df['Pred_class'] = y_pred
    pred_df['Correct_class'] = y_correct
    print(pred_df)

