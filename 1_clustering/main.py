import Beer
from clustering import clustering2dim, clusteringNdim


if __name__ == '__main__':
    # Параметры
    CLUSTERING2DIM = not True
    SHOW_BEST_EPS = not False
    SHOW_ELBOW = not False
    DBSCAN_EPS_2 = 0.03
    DBSCAN_MS_2 = 5
    KMEANS_CN_2 = 5
    DBSCAN_EPS_4 = 0.03
    DBSCAN_MS_4 = 5
    KMEANS_CN_4 = 4

    # Получаем пиво
    beers = Beer.read_beers()

    if CLUSTERING2DIM:
        clustering2dim(beers, SHOW_BEST_EPS, SHOW_ELBOW, DBSCAN_EPS_2, DBSCAN_MS_2, KMEANS_CN_2)
    else:
        clusteringNdim(beers, SHOW_BEST_EPS, SHOW_ELBOW, DBSCAN_EPS_4, DBSCAN_MS_4, KMEANS_CN_4)
