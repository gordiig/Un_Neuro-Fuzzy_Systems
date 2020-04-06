import matplotlib.pyplot as plt
import numpy as np
from Beer import Beer
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from math import sqrt
from utils import COLOR_CODES_LIST, COLOR_CODES_NAMES


def __show_plot(pts, xlabel='x', ylabel='y', color='b', fmt='.', legend='', title='', show=True):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(list(map(lambda x: x[0], pts)),
             list(map(lambda x: x[1], pts)),
             fmt,
             c=color,
             label=legend)
    if show:
        plt.show()


def __show_plot_clustered(pts, xlabel='x', ylabel='y', title='Кластеризованные точки'):
    for i, (cluster_num, beer_vct) in enumerate(pts.items()):
        color_idx = i % len(COLOR_CODES_LIST)
        __show_plot(beer_vct, xlabel=xlabel, ylabel=ylabel, color=COLOR_CODES_LIST[color_idx],
                    legend=f'{cluster_num} кластер', title=title, show=False)
        # print(f'Кластер {cluster_num} выделен цветом {COLOR_CODES_NAMES[color_idx]} ({COLOR_CODES_LIST[color_idx]})')
    plt.legend(fontsize='5')
    plt.show()


def __search_best_eps(beer_vcts, min_samples):
    avg_distances = []
    for i, main_pt in enumerate(beer_vcts):
        cur_distances = []
        for j, dist_pt in enumerate(beer_vcts):
            if j == i:
                continue
            cur_distances.append(sqrt((main_pt[0] - dist_pt[0]) ** 2 + (main_pt[1] - dist_pt[1]) ** 2))
        min_ms_distances = []
        for _ in range(min_samples):
            min_ms_distances.append(min(cur_distances))
            cur_distances.remove(min(cur_distances))
        avg_distances.append(sum(min_ms_distances) / len(min_ms_distances))
    plt.title('Поиск лучшего значения eps')
    plt.xlabel('sorted n')
    plt.ylabel('avg dist')
    plt.plot([i for i in range(len(avg_distances))], sorted(avg_distances))
    plt.show()


def __perform_dbscan(data, eps, min_samples) -> Tuple[List[int], int]:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    res = db.fit_predict(data)
    return res, max(res)


def __group_by_clusters(data, clusters) -> Dict[int, List[List[float]]]:
    clustered_beers_vcts = {}
    for cl_num, beer_vct in zip(clusters, data):
        if cl_num in clustered_beers_vcts.keys():
            clustered_beers_vcts[cl_num].append(beer_vct)
        else:
            clustered_beers_vcts[cl_num] = [beer_vct]
    return clustered_beers_vcts


def __search_best_km_cluster_cnt(data, min_clusters=2, max_clusters=10):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(min_clusters, max_clusters))
    visualizer.fit(np.array(data))
    visualizer.show()


def __perform_kmeans(data, cnumber) -> List[int]:
    km = KMeans(n_clusters=cnumber)
    km_res = km.fit_predict(data)
    return km_res


def __pretty_print_clusters(clustered_data: dict, filename: str, title=''):
    with open(filename, 'w') as f:
        # print('\n\n')
        # print('#' * 70)
        # print(title)
        f.write('cnumber,ibu,abv,style,bcity\n')
        for cluster_num, beers_vct in clustered_data.items():
            prefix = 'Шум,' if cluster_num == -1 else f'{cluster_num},'
            for beer_vct in beers_vct:
                f.write(prefix + '{:3.2f},{:0>3.2f},{:>1.5f},{:>1.5f}\n'.format(*beer_vct))
            # print()


def __normalize_values(data: List[List[float]]) -> List[List[float]]:
    ndims = len(data[0])
    norm_data = data
    for i in range(ndims):
        maxx = max([x[i] for x in data])
        minn = min([x[i] for x in data])
        for j in range(len(data)):
            norm_data[j][i] = (data[j][i] - minn) / (maxx - minn)
    return norm_data


def clustering2dim(beers, show_best_eps, show_elbow, dbscan_eps, dbscan_ms, kmeans_cn):
    beers_vcts = list(map(lambda x: x.get_as_vector_for_clustering_2dim(), beers))
    beers_vcts = __normalize_values(beers_vcts)
    __show_plot(beers_vcts, xlabel='ibu', ylabel='abv', title='Все пивы')

    # Ищем лучшее значение eps для DBSCAN
    if show_best_eps:
        __search_best_eps(beers_vcts, dbscan_ms)
    # DBSCAN
    res, dbscan_clusters_num = __perform_dbscan(beers_vcts, eps=dbscan_eps, min_samples=dbscan_ms)
    print(f'DBSCAN total clusters = {dbscan_clusters_num}')
    # Разносим по кластерам beer_vcts
    clustered_beers_vcts = __group_by_clusters(beers_vcts, res)
    # Визуализируем кластеры
    __show_plot_clustered(clustered_beers_vcts, xlabel='ibu', ylabel='abv', title='Кластеризация DBSCAN')

    # Метод локтя
    if show_elbow:
        __search_best_km_cluster_cnt(beers_vcts)
    # KMeans
    km_res = __perform_kmeans(beers_vcts, kmeans_cn)
    # Разносим по кластерам beer_vcts
    km_beer_vcts_clusters = __group_by_clusters(beers_vcts, km_res)
    # Визуализируем кластеры
    __show_plot_clustered(km_beer_vcts_clusters, xlabel='ibu', ylabel='abv', title='Кластеризация KMeans')


def clusteringNdim(beers: List[Beer], show_best_eps, show_elbow, dbscan_eps, dbscan_ms, kmeans_cn):
    beers_vcts = [beer_vct.get_as_vector_for_clustering() for beer_vct in beers]
    beers_vcts = __normalize_values(beers_vcts)
    # Ищем лучшее значение eps для DBSCAN
    if show_best_eps:
        __search_best_eps(beers_vcts, dbscan_ms)
    # DBSCAN
    res_db, db_clusters_cnt = __perform_dbscan(beers_vcts, dbscan_eps, dbscan_ms)
    print(f'DBSCAN cluster num = {db_clusters_cnt}')
    # Разносим по кластерам beer_vcts
    clustered_beers_vcts_db = __group_by_clusters(beers_vcts, res_db)
    # Вывод пив по кластерам
    __pretty_print_clusters(clustered_beers_vcts_db, filename='dbscan_res.csv', title='DBSCAN')

    # Метод локтя
    if show_elbow:
        __search_best_km_cluster_cnt(beers_vcts)
    # KMeans
    res_km = __perform_kmeans(beers_vcts, kmeans_cn)
    # Разносим по кластерам beers_vcts
    clustered_beers_vcts_km = __group_by_clusters(beers_vcts, res_km)
    # Вывод пив по кластерам
    __pretty_print_clusters(clustered_beers_vcts_km, filename='kmeans_res.csv', title='KMeans')
