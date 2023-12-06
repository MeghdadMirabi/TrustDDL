def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()
