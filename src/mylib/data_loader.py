from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def load_data(dataset_path):
    """
    Загружает данные из файла и подготавливает их для обучения модели.

    Args:
        dataset_path (str): Путь к файлу данных.

    Returns:
        tuple: разделенные на обучающую и тестовую выборку признаки и метки.
    """
    data = load_svmlight_file(dataset_path)
    X, y = data[0].toarray(), data[1]
    y = 2 * y - 3 
    return train_test_split(X, y, test_size=0.2)
