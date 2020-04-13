import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


def plot_PCA(pca_coord):
    pca = PCA(n_components=2)
    show_reduced = pca.fit_transform(pca_coord)
    print()
    print(f'Изначальная размерность: {pca_coord.shape}')
    print(f'Уменьшенная размерность: {show_reduced.shape}')
    fig, ax = plt.subplots()  # Рисуем двумерную картинку.
    ax.scatter(show_reduced[:, 0], show_reduced[:, 1], edgecolor='b')
    ax.set_title('PCA')
    plt.show()


def plot_MDS(mds_coord):
    mds = MDS(random_state=228)
    show_reduced = mds.fit_transform(mds_coord)
    fig, ax = plt.subplots()
    ax.scatter(show_reduced[:, 0], show_reduced[:, 1], edgecolor='b',)
    ax.set_title('MDS')
    plt.show()


def plot_TSNE(tsne_coord):
    tsne = TSNE()
    show_reduced = tsne.fit_transform(tsne_coord)
    fig, ax = plt.subplots()
    ax.scatter(show_reduced[:, 0], show_reduced[:, 1], edgecolor='b', )
    ax.set_title('t-SNE')
    plt.show()


def plot_UMAP(umap_coord):
    umap_emb = umap.UMAP()
    show_reduced = umap_emb.fit_transform(umap_coord)
    fig, ax = plt.subplots()
    ax.scatter(show_reduced[:, 0], show_reduced[:, 1], edgecolor='b', )
    ax.set_title('UMAP')
    plt.show()


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

    # PCA
    plot_PCA(X)

    # MDS
    plot_MDS(X)

    # TSNE
    plot_TSNE(X)

    # UMAP
    plot_UMAP(X)
