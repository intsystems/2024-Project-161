import numpy as np


def loss(w, X, y, lambda_value):
    """
    Вычисляет значение функции потерь логистической регрессии с L2-регуляризацией.

    Args:
        w (np.array): Вектор весов модели.
        X (np.array): Матрица признаков.
        y (np.array): Вектор меток.
        lambda_value (float): Коэффициент регуляризации.

    Returns:
        float: Значение функции потерь.
    """
    yXw = y * (X @ w)
    log_loss = np.mean(np.log(1 + np.exp(-yXw)))
    reg_loss = (lambda_value / 2) * np.dot(w, w)
    return log_loss + reg_loss


def loss_grad(w, X, y, lambda_value):
    """
    Вычисляет градиент функции потерь логистической регрессии.

    Args:
        w (np.array): Вектор весов модели.
        X (np.array): Матрица признаков.
        y (np.array): Вектор меток.
        lambda_value (float): Коэффициент регуляризации.

    Returns:
        np.array: Градиент функции потерь по вектору весов.
    """
    yXw = y * (X @ w)
    log_grad = np.mean((-y * X.T) / (1 + np.exp(yXw)), axis=1)
    reg_grad = lambda_value * w
    return log_grad + reg_grad


class Func:
    """
    Класс для функций, используемых в градиентном спуске.

    Args:
        func (callable): Функция потерь.
        grad (callable): Функция для вычисления градиента.
    """

    def __init__(self, func, grad):
        self.f_ = func
        self.grad = grad

    def __call__(self, x):
        return self.f_(x)


def create_worker_func(f: Func, *args):
    """
    Создает функцию для рабочего, которая будет использовать данные для вычисления градиента.

    Args:
        f (Func): Функция, содержащая методы вычисления значения и градиента.
        *args: Аргументы, которые будут переданы функции `f`.

    Returns:
        Func: Новая функция, адаптированная под конкретные данные.
    """
    def wrapped_f(w):
        return f(w, *args)

    def wrapped_grad(w):
        return f.grad(w, *args)

    return Func(wrapped_f, wrapped_grad)


f = Func(
    func=loss,
    grad=loss_grad
)
