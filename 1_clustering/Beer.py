import csv
from typing import List, Union


class BeerReadError(Exception):
    """
    Класс-ошибка для чтения пива
    """
    def __init__(self, message: str):
        self.message = message


class Beer:
    """
    Класс для пива
    """
    def __init__(self, _id: int, abv: Union[float, str], ibu: Union[float, str], name: str, style: str,
                 ounces: Union[float, str], brewery_name: Union[str, None], brewery_city: Union[str, None],
                 brewery_state: Union[str, None], brewery_id: Union[int, None]):
        self.id = _id
        try:
            self.abv = float(abv)
        except ValueError:
            self.abv = 0
        try:
            self.ibu = float(ibu)
        except ValueError:
            self.ibu = -1
        self.name = name
        self.style = style
        try:
            self.ounces = float(ounces)
        except ValueError:
            self.ounces = 0
        self.brewery_id = brewery_id
        self.brewery_name = brewery_name
        self.brewery_city = brewery_city
        self.brewery_state = brewery_state

    def get_as_vector_for_clustering_2dim(self):
        return [self.ibu, self.abv]

    def get_as_vector_for_clustering(self):
        return [self.ibu, self.abv, hash(self.style), hash(self.brewery_city)]

    def repr_with_cluster_attrs(self):
        return f'id = {self.id} | ibu = {self.ibu} | abv = {self.abv} | style = {self.style} ' \
               f'| bcity = {self.brewery_city}'

    def __str__(self):
        return f'Beer: id = {self.id} | name = {self.name} | bid = {self.brewery_id} | bname = {self.brewery_name}'

    def __repr__(self):
        return self.__str__()


def read_beers(beerfilename='beers.csv', breweriyfilename='breweries.csv') -> List[Beer]:
    """
    Чтение пив из файла
    :param beerfilename: Путь к файлу пив
    :param breweriyfilename: Путь к файлу пивоварен
    :return: Список пив
    """
    # Чтение файлов
    try:
        beers_rows = __read_csv(beerfilename)[1:]
    except (OSError, IOError):
        raise BeerReadError('Не получается считать файл с пивом')
    try:
        breweries_rows = __read_csv(breweriyfilename)[1:]
    except (OSError, IOError):
        raise BeerReadError('Не получается считать файл с пивоварнями')
    # Формирование массива пив
    ans = []
    for beer_row in beers_rows:
        brewery_row = list(filter(lambda x: x[0] == beer_row[6], breweries_rows))
        brewery_row = [None] * 4 if len(brewery_row) == 0 else brewery_row[0]
        beer = Beer(_id=int(beer_row[3]), abv=beer_row[1], ibu=beer_row[2], name=beer_row[4], style=beer_row[5],
                    ounces=beer_row[7], brewery_name=brewery_row[1], brewery_city=brewery_row[2],
                    brewery_state=brewery_row[3], brewery_id=brewery_row[0])
        ans.append(beer)
    return ans


def __read_csv(filename: str) -> List[List[str]]:
    """
    Читает csv-файл в массив строк
    :param filename: Имя файла
    :return: Массив строк
    """
    ans = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            ans.append(row)
    return ans
