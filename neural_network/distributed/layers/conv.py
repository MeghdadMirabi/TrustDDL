from fixedpoint.fp_util import mul, to_config_representation, div
from neural_network.distributed.layers import Layer


class Convolution(Layer):
    """
    A layer that applies a convolution to the input.
    """

    _auxiliary_nodes: []
    _group_num: int

    def __init__(self, batch_size, lr, F, b, F_dim, in_dim, output_dim, padding, stride, calc_partials, empty_like, empty, zeros_like, pad, dot, auxiliary_nodes, group_num, node_id, byzantine) -> None:
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
        auxiliary_nodes:
            A list of :class:`actors.AuxiliaryNode` ray stubs.
        group_num: int
            The group number of the node using this network (used during secure computations).
        """

        super().__init__(batch_size, lr, byzantine, group_num, node_id)
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
        self._auxiliary_nodes = auxiliary_nodes
        self._group_num = group_num
        self.calc_partials = calc_partials

    async def forward(self, X, batch):
        """
        Applies the convolution operation to the input ``X``, using the filters ``F`` and bias ``b``.
        Returns the result of the convolution plus the bias.

        Parameters
        ----------
        X:
            A share of the 3D input tensor over which to convolve. The input will not be padded.

        Returns
        -------
        conv_result:
            A share of the result of the convolution plus the bias as a 3D tensor obtained by sliding each filter in ``F`` across the
            input data, computing the dot product at each location and adding the bias for each filter. The result of each
            filter application is 2-dimensional and will be stacked on the along the channel dimension.
        """

        if batch:
            X = [[self._pad(__X, (self.padding_width, self.padding_width, self.padding_height, self.padding_height)) for __X in _X] for _X in X]
        else:
            X = [self._pad(_X, (self.padding_width, self.padding_width, self.padding_height, self.padding_height)) for _X in X]

        self._input = X

        if batch:
            self._output = [[self._empty(self.output_dim) for _ in _input] for _input in self._input]
        else:
            self._output = [self._empty(self.output_dim) for _ in self._input]

        for m in range(self.output_dim[1]):
            for n in range(self.output_dim[2]):
                filter_res = await self._apply_filter(m, n, X, batch)

                if batch:
                    for i, _filter_res in enumerate(filter_res):
                        for j in range(len(_filter_res)):
                            self._output[i][j][:, m, n] = _filter_res[j]
                else:
                    for i, _filter_res in enumerate(filter_res):
                        self._output[i][:, m, n] = _filter_res

        if batch:
            output = []
            for _output in self._output:  # iterate over batch
                _out_batch = []
                for __output, _b in zip(_output, self.b):  # iterate over shares to maintain
                    _out_shares = __output
                    for j in range(self.output_dim[0]):  # iterate over output channels
                        _out_shares[j] += _b[j]
                    _out_batch.append(_out_shares)
                output.append(_out_batch)
        else:
            output = []
            for _output, _b in zip(self._output, self.b):  # iterate over shares to maintain
                _out_shares = _output
                for j in range(self.output_dim[0]):  # iterate over output channels
                    _out_shares[j] += _b[j]
                output.append(_out_shares)

        self._output = output

        return self._output

    async def _apply_filter(self, m, n, X, batch):
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
            X_part = [[__X[:, m * self.stride_height:m * self.stride_height + self.F_dim[2], n * self.stride_width:n * self.stride_width + self.F_dim[3]] for __X in _X] for _X in X]
            F = [self.F for _ in X]
        else:
            X_part = [_X[:, m * self.stride_height:m * self.stride_height + self.F_dim[2], n * self.stride_width:n * self.stride_width + self.F_dim[3]] for _X in X]
            F = self.F

        mult = await self.sec_comp.sec_mul(F, X_part, self.replicas_to_maintain, batch=batch, commit=(m == 0 and n == 0))

        if batch:
            result = [[__mult.sum([1, 2, 3]) for __mult in _mult] for _mult in mult]
        else:
            result = [_mult.sum([1, 2, 3]) for _mult in mult]

        return result

    async def backward(self, partials_prev, batch):
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

        if batch:
            partials_prev = [[__partials_prev.reshape(self.output_dim) for __partials_prev in _partials_prev] for _partials_prev in partials_prev]
            new_partials = [[self._empty(self.in_dim) for __input in _input] for _input in self._input]
        else:
            partials_prev = [_partials_prev.reshape(self.output_dim) for _partials_prev in partials_prev]
            new_partials = [self._empty(self.in_dim) for _input in self._input]

        new_gradient = [self._zeros_like(_F) for _F in self.F]

        for m in range(self.F_dim[1]):
            for n in range(self.F_dim[2]):
                for o in range(self.F_dim[3]):
                    updates = await self._calc_update(m, n, o, partials_prev, batch)
                    for i in range(len(updates)):
                        new_gradient[i][:, m, n, o] = updates[i]

        if self.calc_partials:
            height_slide_indexes = [i * self.stride_height - self.padding_height for i in range(self.output_dim[1])]
            width_slide_indexes = [i * self.stride_width - self.padding_width for i in range(self.output_dim[2])]
            for m in range(self.in_dim[1]):
                for n in range(self.in_dim[2]):
                    partial = await self._calc_partial(m, n, partials_prev, height_slide_indexes, width_slide_indexes, batch)

                    if batch:
                        for i in range(len(partial)):
                            for j in range(len(partial[i])):
                                new_partials[i][j][:, m, n] = partial[i][j]
                    else:
                        for i in range(len(partial)):
                            new_partials[i][:, m, n] = partial[i]

        F_update = new_gradient
        if batch:
            b_update = [sum([_partials_prev[i] for _partials_prev in partials_prev]) for i in range(len(self.replicas_to_maintain))]
        else:
            b_update = partials_prev

        if batch:
            self.F = [_F - div(mul(to_config_representation(self._lr, scalar=True, value_type='float'), _F_update), to_config_representation(len(partials_prev), scalar=True, value_type='int')) for _F, _F_update in zip(self.F, F_update)]
            self.b = [_b - div(mul(to_config_representation(self._lr, scalar=True, value_type='float'), _b_update.sum([1, 2])), to_config_representation(len(partials_prev), scalar=True, value_type='int')) for _b, _b_update in zip(self.b, b_update)]
        else:
            self.F = [_F - mul(to_config_representation(self._lr, scalar=True, value_type='float'), _F_update) for _F, _F_update in zip(self.F, F_update)]
            self.b = [_b - mul(to_config_representation(self._lr, scalar=True, value_type='float'), _b_update.sum([1, 2])) for _b, _b_update in zip(self.b, b_update)]

        return new_partials

    async def _calc_update(self, m, n, o, partials_prev, batch):
        """
        Calculates the gradient of the filters and bias with respect to the loss for one position of the layer's input.
        """

        in_part = []
        if batch:
            for _input in self._input:
                _in_part = []
                for __input in _input:
                    _inputs = __input[m]
                    _inputs = _inputs[[n + self.stride_height * i for i in range(self.output_dim[1])]]
                    _inputs = _inputs[:, [o + self.stride_width * i for i in range(self.output_dim[2])]]
                    _in_part.append(_inputs)
                in_part.append(_in_part)
        else:
            for _input in self._input:
                _inputs = _input[m]
                _inputs = _inputs[[n + self.stride_height * i for i in range(self.output_dim[1])]]
                _inputs = _inputs[:, [o + self.stride_width * i for i in range(self.output_dim[2])]]
                in_part.append(_inputs)

        mult = await self.sec_comp.sec_mul(in_part, partials_prev, self.replicas_to_maintain, batch=batch, commit=(m == 0 and n == 0 and o == 0))

        if batch:
            result = [sum([_mult[i].sum([1, 2]) for _mult in mult]) for i in range(len(self.replicas_to_maintain))]
        else:
            result = [_mult.sum([1, 2]) for _mult in mult]

        return result

    async def _calc_partial(self, m, n, partials_prev, height_slide_indexes, width_slide_indexes, batch):
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
            _F_part = []
            for _F in self.F:
                __F_part = _F[:, :, F_height_idxs]
                __F_part = __F_part[:, :, :, F_width_idxs]
                _F_part.append(__F_part)
            F_part = [_F_part for _ in partials_prev]

            partials_prev_part = []
            for _partials_prev in partials_prev:
                _partials_prev_parts = []
                for __partials_prev in _partials_prev:
                    partials_part = __partials_prev[:, partials_height_idxs]
                    partials_part = partials_part[:, None, :, partials_width_idxs]
                    _partials_prev_parts.append(partials_part)
                partials_prev_part.append(_partials_prev_parts)
        else:
            F_part = []
            for _F in self.F:
                _F_part = _F[:, :, F_height_idxs]
                _F_part = _F_part[:, :, :, F_width_idxs]
                F_part.append(_F_part)

            partials_prev_part = []
            for _partials_prev in partials_prev:
                partials_part = _partials_prev[:, partials_height_idxs]
                partials_part = partials_part[:, None, :, partials_width_idxs]
                partials_prev_part.append(partials_part)

        mult = await self.sec_comp.sec_mul(F_part, partials_prev_part, self.replicas_to_maintain, batch=batch, commit=(m == 0 and n == 0))

        if batch:
            result = [[__mult.sum([0, 2, 3]) for __mult in _mult] for _mult in mult]
        else:
            result = [_mult.sum([0, 2, 3]) for _mult in mult]

        return result

    def update_parameters(self):
        """
        Update the parameters of the layer based on the accumulated gradients.
        """
        pass

    def clone(self, group_num, node_id, byzantine):
        """
        Create a clone of this layer for the given group number ``group_num``.

        Parameters
        ----------
        group_num: int
            The group number to create the copy for.

        Returns
        -------
        clone:
            A clone of this layer for the given group number ``group_num``. The returned layer has the same
            attributes (batch size, learning rate, ...) as this layer, except for the group number. Secret-shared
            parameters (filters, biases) are not copied as they will be overwritten by other secret shares.
        """

        return Convolution(self._batch_size, self._lr, None, None, self.F_dim, self.in_dim, self.output_dim, (self.padding_height, self.padding_width),
                           (self.stride_height, self.stride_width), self.calc_partials, self._empty_like, self._empty, self._zeros_like, self._pad,
                           self._dot, self._auxiliary_nodes, group_num, node_id, byzantine)
