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
    def _pad_neg_inf(tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        return np.pad(tensor, ((0, 0), (0, 0), \
            (one_size_pad, one_size_pad), (one_size_pad, one_size_pad)), \
                 mode='constant', constant_values=-np.inf)

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[-1] == input.shape[-2]
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        result = np.zeros((input.shape[0], 
            input.shape[1], 
            (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        ))

        padded_input = self._pad_neg_inf(input, self.padding)

        self.grad_poses = np.zeros_like(input)

        for b in range(result.shape[0]):
            for c in range(result.shape[1]):
                for i in range(result.shape[2]):
                    for j in range(result.shape[3]):
                        result[b, c, i, j] = \
                            np.max(padded_input[b, c, \
                                i * self.stride: i * self.stride + self.kernel_size, \
                                    j * self.stride: j * self.stride + self.kernel_size])

                        ind_max = np.unravel_index(np.argmax(padded_input[b, c, \
                                i * self.stride: i * self.stride + self.kernel_size, \
                                    j * self.stride: j * self.stride + self.kernel_size]), \
                                        (self.kernel_size, self.kernel_size))

                        self.grad_poses[b, c, \
                                -self.padding + i * self.stride + ind_max[0], \
                                    -self.padding + j * self.stride + ind_max[1]] = 1

        return result

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        result = np.zeros_like(self.grad_poses)
        for b in range(result.shape[0]):
            for c in range(result.shape[1]):
                i_out = 0
                j_out = 0
                for i in range(result.shape[2]):
                    for j in range(result.shape[3]):
                        if self.grad_poses[b, c, i, j] == 1:
                            result[b, c, i, j] = output_grad[b, c, i_out, j_out]
                            j_out += 1
                            if j_out  == output_grad.shape[3]:
                                i_out += 1
                                j_out = 0


        return result

