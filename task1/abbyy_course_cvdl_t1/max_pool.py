import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        

    @staticmethod
    def _pad_neg_inf(tensor: np.ndarray, one_size_pad: int, axis=[-1, -2]) -> np.ndarray:
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        npad = np.array([(0, 0)] * tensor.ndim)
        npad[axis] = (one_size_pad, one_size_pad)
        return np.pad(tensor, npad, mode='constant', constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        result = np.zeros((input.shape[0], 
            input.shape[1], 
            (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        ))

        self.size = input.shape
        padded_input = self._pad_neg_inf(input, self.padding)

        self.grad_poses = np.zeros(result.shape + tuple([2]), dtype=int)
        
        for b in range(result.shape[0]):
            for c in range(result.shape[1]):
                for i in range(result.shape[2]):
                    for j in range(result.shape[3]):
                        ind_max = np.unravel_index(np.argmax(padded_input[b, c, \
                                i * self.stride: i * self.stride + self.kernel_size, \
                                    j * self.stride: j * self.stride + self.kernel_size]), \
                                        (self.kernel_size, self.kernel_size))

                        value = padded_input[b, c, \
                                i * self.stride + ind_max[0], \
                                    j * self.stride + ind_max[1]]

                        result[b, c, i, j] = value
                        
                        self.grad_poses[b, c, i, j, :] = \
                            [-self.padding + i * self.stride + ind_max[0], \
                                    -self.padding + j * self.stride + ind_max[1]]

        return result

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        result = np.zeros(self.size)
        for b in range(self.grad_poses.shape[0]):
            for c in range(self.grad_poses.shape[1]):
                for i in range(self.grad_poses.shape[2]):
                    for j in range(self.grad_poses.shape[3]):
                        idx = self.grad_poses[b, c, i, j]
                        result[b, c, idx[0], idx[1]] += output_grad[b, c, i, j]

        return result

