from neural_network.centralized.layers import Layer


class Convolution(Layer):
    """
    A layer that applies a convolution to the input.
    """

    def __init__(self, batch_size, lr, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials, empty_like, empty, zeros_like, pad, dot) -> None:
        """
        Parameters
        ----------
        batch_size: int
            The number of samples in a batch.
        lr: float
            The learning rate for the layer.
        F:
            The filters for the layer.
        b:
            The biases for the layer.
        F_dim:
            The dimensions of the filters. F_dim[0] is the number of filters and F_dim[1:4] are the filter dimensions.
        in_dim:
            The dimensions of the input.
        output_dim:
            The dimensions of the output.
        empty_like: callable
            A function that returns an empty tensor like the input.
        empty: callable
            A function that returns an empty tensor.
        zeros_like: callable
            A function that returns a tensor of zeros like the input.
        dot: callable
            A function that returns the dot product of two tensors.
        """

        super().__init__(batch_size, lr)
        self.F = F
        self.b = b
        self.F_dim = F_dim  # F_dim[0]: number of filters, F_dim[1:4]: filter dimension
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.padding = padding
        self.stride = stride
        self.padding_height, self.padding_width = padding
        self.stride_height, self.stride_width = stride
        self._output = None
        self._empty_like = empty_like
        self._empty = empty
        self._zeros_like = zeros_like
        self._pad = pad
        self._dot = dot
        self._input = None
        self._gradient = None
        self.calc_partials = calc_partials

    def forward(self, X):
        """
        Applies the convolution operation to the input ``X``, using the filters ``F`` and bias ``b``.
        Returns the result of the convolution plus the bias.

        Parameters
        ----------
        X:
            The 3D input tensor over which to convolve. The input will not be padded.

        Returns
        -------
        conv_result:
            The result of the convolution plus the bias as a 3D tensor obtained by sliding each filter in ``F`` across the
            input data, computing the dot product at each location and adding the bias for each filter. The result of each
            filter application is 2-dimensional and will be stacked on the along the channel dimension.
        """

        batch = type(X) == list

        if batch:
            X = [self._pad(_X, (self.padding_width, self.padding_width, self.padding_height, self.padding_height)) for _X in X]
        else:
            X = self._pad(X, (self.padding_width, self.padding_width, self.padding_height, self.padding_height))

        self._input = X

        if batch:
            self._output = [self._empty(self.output_dim) for _ in self._input]
        else:
            self._output = self._empty(self.output_dim)

        for m in range(self.output_dim[1]):
            for n in range(self.output_dim[2]):
                filter_res = self._apply_filter(m, n, X, batch)

                if batch:
                    for i, _filter_res in enumerate(filter_res):
                        self._output[i][:, m, n] = _filter_res
                else:
                    self._output[:, m, n] = filter_res

        if batch:
            for i in range(len(self._output)):
                for j in range(self.output_dim[0]):
                    self._output[i][j] += self.b[j]
        else:
            for i in range(self.output_dim[0]):
                self._output[i] += self.b[i]

        return self._output

    def _apply_filter(self, m, n, X, batch):
        """
        Applies a filter to a region of the input.

        Parameters
        ----------
        m: int
            Offset of the region of the input in the second dimension.
        n: int
            Offset of the region of the input in the third dimension.
        X: array_like
            The input data.

        Returns
        -------
        array_like
            The result of applying the filter to the input at the specified region.
        """

        if batch:
            result = []
            for _X in X:
                mult = self.F * _X[:, m * self.stride_height:m * self.stride_height + self.F_dim[2], n * self.stride_width:n * self.stride_width + self.F_dim[3]]
                result.append(mult.sum([1, 2, 3]))
        else:
            X_part = X[:, m * self.stride_height:m * self.stride_height + self.F_dim[2], n * self.stride_width:n * self.stride_width + self.F_dim[3]]
            mult = self.F * X_part
            result = mult.sum([1, 2, 3])

        return result

    def backward(self, partials_prev):
        """
        Computes the gradients of the filters and bias with respect to the loss, given the
        partial derivatives of the loss with respect to the output of the layer.
        Updates the filters and bias using the computed gradients.
        Returns the partial derivatives of the loss with respect to the input of the layer.

        Parameters
        ----------
        partials_prev:
            The partial derivatives of the loss with respect to the output.

        Returns
        -------
        new_partials:
            The partial derivatives of the loss with respect to the input.
        """

        batch = type(partials_prev) == list

        if batch:
            partials_prev = [_partials_prev.reshape(self.output_dim) for _partials_prev in partials_prev]
            new_partials = [self._empty(self.in_dim) for _input in self._input]
        else:
            partials_prev = partials_prev.reshape(self.output_dim)
            new_partials = self._empty(self.in_dim)

        new_gradient = self._zeros_like(self.F)

        for m in range(self.F_dim[1]):
            for n in range(self.F_dim[2]):
                for o in range(self.F_dim[3]):
                    new_gradient[:, m, n, o] = self._calc_update(m, n, o, partials_prev, batch)

        if self.calc_partials:
            height_slide_indexes = [i * self.stride_height - self.padding_height for i in range(self.output_dim[1])]
            width_slide_indexes = [i * self.stride_width - self.padding_width for i in range(self.output_dim[2])]
            for m in range(self.in_dim[1]):
                for n in range(self.in_dim[2]):
                    partial = self._calc_partial(m, n, partials_prev, height_slide_indexes, width_slide_indexes, batch)
                    if batch:
                        for i in range(len(partial)):
                            new_partials[i][:, m, n] = partial[i]
                    else:
                        new_partials[:, m, n] = partial

        F_update = new_gradient
        b_update = sum(partials_prev).sum([1, 2]) if batch else partials_prev.sum([1, 2])

        if batch:
            self.F -= self._lr * F_update / len(partials_prev)
            self.b -= self._lr * b_update / len(partials_prev)
        else:
            self.F -= self._lr * F_update
            self.b -= self._lr * b_update

        return new_partials if self.calc_partials else None

    def _calc_update(self, m, n, o, partials_prev, batch):
        """
        Calculates the gradient of the filters and bias with respect to the loss for one position of the layer's input.
        """

        if batch:
            result = []
            for _input, _partials_prev in zip(self._input, partials_prev):
                _inputs = _input[m]
                _inputs = _inputs[[n + self.stride_height * i for i in range(self.output_dim[1])]]
                _inputs = _inputs[:, [o + self.stride_width * i for i in range(self.output_dim[2])]]
                mult = _inputs * _partials_prev
                result.append(mult.sum([1, 2]))
            result = sum(result)
        else:
            _inputs = self._input[m]
            _inputs = _inputs[[n+self.stride_height*i for i in range(self.output_dim[1])]]
            _inputs = _inputs[:, [o+self.stride_width*i for i in range(self.output_dim[2])]]
            mult = _inputs * partials_prev
            result = mult.sum([1, 2])

        return result

    def _calc_partial(self, m, n, partials_prev, height_slide_indexes, width_slide_indexes, batch):
        """
        Calculates the partial derivative of the loss with respect to the input of the layer at one position.
        """

        F_height_min = max(m - height_slide_indexes[-1], m % self.stride_height)
        F_height_max = min(self.F_dim[2], m + self.padding_height + 1)
        F_width_min = max(n - width_slide_indexes[-1], n % self.stride_width)
        F_width_max = min(self.F_dim[3], n + self.padding_width + 1)

        F_height_idxs = list(range(F_height_min, F_height_max, self.stride_height))
        F_width_idxs = list(range(F_width_min, F_width_max, self.stride_width))

        partials_height_idxs = [i for i, slide_idx in enumerate(height_slide_indexes) if slide_idx <= m < slide_idx + self.F_dim[2]]
        partials_width_idxs = [i for i, slide_idx in enumerate(width_slide_indexes) if slide_idx <= n < slide_idx + self.F_dim[3]]

        if batch:
            result = []

            F_part = self.F[:, :, F_height_idxs]
            F_part = F_part[:, :, :, F_width_idxs]

            for _partials_prev in partials_prev:
                partials_part = _partials_prev[:, partials_height_idxs]
                partials_part = partials_part[:, None, :, partials_width_idxs]

                mult = F_part * partials_part
                result.append(mult.sum([0, 2, 3]))
        else:
            F_part = self.F[:, :, F_height_idxs]
            F_part = F_part[:, :, :, F_width_idxs]

            partials_part = partials_prev[:, partials_height_idxs]
            partials_part = partials_part[:, None, :, partials_width_idxs]

            mult = F_part * partials_part

            result = mult.sum([0, 2, 3])

        return result

    def update_parameters(self):
        """
        Update the parameters of the layer based on the accumulated gradients.
        """
        pass
