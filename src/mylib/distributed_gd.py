import numpy as np

class Worker:
    """
    Класс рабочего, который выполняет вычисления градиента.

    Args:
        func (Func): Функция, содержащая методы вычисления значения и градиента.
    """
    def __init__(self, func):
        self.f = func
        self.w = None
        self.compress_op = None
        self.error = 0

    def get_gradient(self):
        return self.compress_op(self.f.grad(self.w))
    
    def get_gradient_ef21(self):
        c = self.compress_op(self.f.grad(self.w) - self.error)
        self.error += c
        return c

class DistributedGD:
    """
    Класс для распределенного градиентного спуска.

    Args:
        workers (list of Worker): Список рабочих, каждый из которых вычисляет градиент.
        step (float or callable): Шаг обучения или функция для его вычисления.
    """
    def __init__(self, workers, step):
        self.workers = workers
        if isinstance(step, (int, float)):
            self.step = lambda *_: step
        else:
            self.step = step
        self.history = []

    def reset(self):
        self.history = []
        for worker in self.workers:
            worker.w = None
            worker.error = 0

    def run(self, num_iter, w0):
        for worker in self.workers:
            worker.w = w0
        w = w0
        for k in range(num_iter):
            mean_grad = np.mean([worker.get_gradient() for worker in self.workers], axis=0)
            w = w - self.step(w, k) * mean_grad

            for _, worker in enumerate(self.workers):
                worker.w = w

            self.history.append(w)

    def run_ef21(self, num_iter, w0):
        for worker in self.workers:
            worker.w = w0
        w = w0
        mean_grad = 0
        for k in range(num_iter):
            mean_add = np.mean([worker.get_gradient_ef21() for worker in self.workers], axis=0)
            mean_grad += mean_add
            w = w - self.step(w, k) * mean_grad

            for _, worker in enumerate(self.workers):
                worker.w = w

            self.history.append(w)