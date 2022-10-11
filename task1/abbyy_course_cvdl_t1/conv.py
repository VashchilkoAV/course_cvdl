import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.parameters.append(np.zeros((out_channels, in_channels, kernel_size, kernel_size)))
        self.parameters.append(np.zeros((out_channels)))
        self.parameters_grads.append(np.zeros_like(self.parameters[0]))
        self.parameters_grads.append(np.zeros_like(self.parameters[1]))

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        npad = np.array([(0, 0)] * tensor.ndim)
        npad[axis] = (one_side_pad, one_side_pad)
        return np.pad(tensor, npad, mode='constant', constant_values=0)

    @staticmethod
    def _cross_correlate(input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1

        offset = (kernel.shape[-1] - 1) // 2
        result = np.zeros((input.shape[0], kernel.shape[0], input.shape[2], input.shape[3]))

        padded_input = ConvLayer._pad_zeros(input, offset).copy()
        
        print(input.shape, kernel.shape)

        for b in range(result.shape[0]):
            for c_out in range(result.shape[1]):
                for i in range(result.shape[2]):
                    for j in range(result.shape[3]):
                        window = padded_input[b, :, i: i + kernel.shape[-1], j: j + kernel.shape[-1]]
                        res = (window * kernel[c_out]).sum()
                        result[b, c_out, i, j] += res

        return result

    def forward(self, input: np.ndarray) -> np.ndarray:
        offset = (self.kernel_size - 1) // 2
        self.prev_input = input.copy()
        padded_input = ConvLayer._pad_zeros(input, offset).copy()
        result = np.zeros((input.shape[0], self.out_channels, input.shape[2], input.shape[3]))

        for b in range(result.shape[0]):
            for c in range(result.shape[1]):
                for i in range(result.shape[2]):
                    for j in range(result.shape[3]):
                        window = padded_input[b, :, i: i + self.kernel_size, j: j + self.kernel_size]
                        res = (window * self.parameters[0][c]).sum() + self.parameters[1][c]
                        result[b, c, i, j] += res

        #print(self.parameters[1])
        #print(np.array([i * np.ones((input.shape[-1], input.shape[-2])) for i in self.parameters[1]])[None, ...].shape)
        return ConvLayer._cross_correlate(input, self.parameters[0]) + np.array([i * np.ones((input.shape[-1], input.shape[-2])) for i in self.parameters[1]])[None, ...]


    def backward(self, output_grad: np.ndarray)->np.ndarray:
        offset = (self.kernel_size - 1) // 2
        padded_input = ConvLayer._pad_zeros(self.prev_input, offset).copy()
        padded_result = np.zeros_like(ConvLayer._pad_zeros(self.prev_input, offset))
        for b in range(output_grad.shape[0]):
            for c in range(self.parameters_grads[0].shape[0]):
                for i in range(self.prev_input.shape[2]):
                    for j in range(self.prev_input.shape[3]):
                        padded_result[b, :, i: i + self.kernel_size, j:j + self.kernel_size] += self.parameters[0][c] * output_grad[b, c, i, j]
                        self.parameters_grads[0][c] += padded_input[b, :, i: i + self.kernel_size, j:j + self.kernel_size] * output_grad[b, c, i, j]

        self.parameters_grads[1] += output_grad.sum(tuple([0, -1, -2]))
        #print(ConvLayer._cross_correlate(self.prev_input, output_grad).shape)
        #print(padded_result[:, :, offset: -offset, offset: -offset], ConvLayer._cross_correlate(self.prev_input, output_grad))
        #self.parameters_grads[0] += ConvLayer._cross_correlate(self.prev_input, output_grad)
        return padded_result[:, :, offset: -offset, offset: -offset]