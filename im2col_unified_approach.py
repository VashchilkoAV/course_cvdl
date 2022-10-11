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
        return np.pad(tensor, npad, mode='constant', constant_values=0).copy()

    @staticmethod
    def _get_im2col_indices(x_shape, field_height, field_width, padding=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape

        out_height = (H + 2 * padding - field_height) + 1
        out_width = (W + 2 * padding - field_width) + 1

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = 1 * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = 1 * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    @staticmethod
    def _im2col_indices(x, field_height, field_width, padding=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = ConvLayer._pad_zeros(x, p)

        k, i, j = ConvLayer._get_im2col_indices(x.shape, field_height, field_width, padding)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    @staticmethod
    def _col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = ConvLayer._get_im2col_indices(x_shape, field_height, field_width, padding)

        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

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
        
        X_col = ConvLayer._im2col_indices(input, kernel.shape[-1], kernel.shape[-2], padding=offset)
        W_col = kernel.reshape(kernel.shape[0], -1)

        out = W_col @ X_col
        out = out.reshape(kernel.shape[0], input.shape[2], input.shape[3], input.shape[0])
        out = out.transpose(3, 0, 1, 2)
        
        return out

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.prev_input = input.copy()
        
        return ConvLayer._cross_correlate(input, self.parameters[0]) + self.parameters[1][:, None, None]

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        offset = (self.kernel_size - 1) // 2

        dout_reshaped = output_grad.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        X_col = ConvLayer._im2col_indices(self.prev_input, self.kernel_size, self.kernel_size, padding=offset)
        dW = dout_reshaped @ X_col.T
        dW = dW.reshape(self.parameters_grads[0].shape)

        W_reshape = self.parameters[0].reshape(self.out_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = ConvLayer._col2im_indices(dX_col, self.prev_input.shape, self.kernel_size, self.kernel_size, padding=offset)

        self.parameters_grads[1] += output_grad.sum(tuple([0, -1, -2]))
        self.parameters_grads[0] += dW

        return dX