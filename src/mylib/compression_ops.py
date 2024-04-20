import numpy as np
from scipy.linalg import svd

def unfold(tensor, mode):
    """
    Разворачивает тензор в матрицу по указанному измерению.

    Args:
        tensor (np.array): Многомерный массив (тензор).
        mode (int): Измерение, по которому происходит разворот.

    Returns:
        np.array: Развернутый тензор в виде матрицы.
    """
    sz = range(len(tensor.shape))
    new_tensor = np.moveaxis(tensor, sz, np.roll(sz, mode))
    return np.reshape(new_tensor, (tensor.shape[mode], -1))

def hosvd_approx(tensor, ranks):
    """
    Выполняет аппроксимацию тензора с помощью высшего сингулярного разложения (HOSVD).

    Args:
        tensor (np.array): Входной тензор.
        ranks (list of int): Список рангов для аппроксимации по каждому измерению.

    Returns:
        np.array: Аппроксимация тензор.
    """
    U_matrices = []
    core_tensor = tensor
    for mode in range(len(tensor.shape)):
        U, _, _ = svd(unfold(tensor, mode), full_matrices=False)
        U_full = np.zeros_like(U)
        min_dim = min(U.shape[1], ranks[mode])
        U_full[:, :min_dim] = U[:, :min_dim]
        U_matrices.append(U_full)
        core_tensor = np.moveaxis(np.tensordot(core_tensor, U_full.T, axes=[mode, 1]), -1, mode)

    tensor = core_tensor
    for mode, U in enumerate(U_matrices):
        tensor = np.moveaxis(np.tensordot(tensor, U, axes=[mode, 1]), -1, mode)

    return tensor

def hosvd_approximation_low_rank_compression(arr, rank):
    """
    Применяет HOSVD для сжатия массива.

    Args:
        arr (np.array): Входной массив.
        rank (int): Желаемый ранг сжатия.

    Returns:
        np.array: Сжатый массив.
    """
    X_matrix = arr.reshape(7, 2, 2, 2, 2)
    x_compressed = hosvd_approx(X_matrix, [1, 1, 2, 1, 1])
    return x_compressed.flatten()

def hosvd_approximation_low_rank_compression_op(rank=1):
    """
    Создает оператор сжатия с использованием HOSVD аппроксимации низкого ранга.

    Args:
        rank (int): Ранг для сжатия.

    Returns:
        function: Оператор сжатия.
    """
    return lambda arr: hosvd_approximation_low_rank_compression(arr, rank)

def top_k(k, arr):
    """
    Оставляет в массиве только k наибольших по модулю элементов, остальные зануляет.

    Args:
        k (int): Количество элементов для сохранения.
        arr (np.array): Входной массив.

    Returns:
        np.array: Модифицированный массив.
    """
    new_arr = arr.copy()
    abs_arr = np.abs(arr)
    new_arr[abs_arr < np.sort(abs_arr)[-k]] = 0
    return new_arr

def top_k_op(k):
    """
    Создает оператор сжатия, который оставляет только k наибольших по модулю элементов массива.

    Args:
        k (int): Количество элементов для сохранения.

    Returns:
        function: Оператор сжатия.
    """
    return lambda arr: top_k(k, arr)

def svd_approximation_low_rank_compression(x, rank=2):
    """
    Применяет SVD для сжатия массива до указанного ранга.

    Args:
        x (np.array): Входной массив.
        rank (int): Желаемый ранг сжатия.

    Returns:
        np.array: Сжатый массив.
    """
    X_matrix = x.reshape(14, 8)
    U, s, Vt = np.linalg.svd(X_matrix, full_matrices=False)
    S = np.diag(s[:rank])
    U_low_rank = U[:, :rank]
    Vt_low_rank = Vt[:rank, :]
    X_approx = np.dot(U_low_rank, np.dot(S, Vt_low_rank))
    return X_approx.reshape(x.shape)

def svd_approximation_low_rank_compression_op(rank=2):
    """
    Создает оператор сжатия с использованием SVD аппроксимации низкого ранга.

    Args:
        rank (int): Ранг для сжатия.

    Returns:
        function: Оператор сжатия.
    """
    return lambda arr: svd_approximation_low_rank_compression(arr, rank)

def identity_op():
    """
    Создает оператор сжатия, который не изменяет данные (тождественное преобразование).

    Returns:
        function: Оператор сжатия.
    """
    return lambda arr: arr
