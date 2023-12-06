from data.preparation.mnist import mnist
from secret_sharing.additive_secret_sharing import create_shares


class DataOwner:

    def __init__(self, num_groups, random_fct, zeros) -> None:
        self.num_groups = num_groups
        self._random_fct = random_fct
        self._zeros = zeros
        self.batch_size = 100
        self.mnist_loaders = mnist.prepare_loaders(batch_size=self.batch_size)
        self.mnist_test_set = None
        self.svhn_test_set = None

    def mnist_train_shares(self, idx):
        """
        Get the ``idx``-th training example of the MNIST data set as additive secret shares.

        Parameters
        ----------
        idx: int
            The index of the training example to return.

        Returns
        -------
        image_shares, label_shares:
            ``(image_shares, label_shares)``, where ``image_shares`` is a list of additive secret shares of the input (image),
            and ``label_shares`` is a list of additive secret shares of the output (one-hot label vector) of the ``idx``-th
            training example. Returns ``None`` if the index is out of bounds for the training set.
        """

        if idx >= len(self.mnist_loaders['train'].dataset):
            return None

        image, label = self.mnist_loaders['train'].dataset.__getitem__(idx)

        label_tensor = self._zeros(10)
        label_tensor[label] = 1

        image_shares = create_shares(image, self.num_groups)
        label_shares = create_shares(label_tensor, self.num_groups)

        return image_shares, label_shares

    def mnist_train_data(self, idx):
        """
        Get the ``idx``-th training example of the MNIST data set as clear text tensors.

        Parameters
        ----------
        idx: int
            The index of the training example to return.

        Returns
        -------
        image_shares, label_shares:
            ``(image_shares, label_shares)``, where ``image_shares`` is the input (image), and ``label_shares`` is
            the output (one-hot label vector) of the ``idx``-th training example. Returns ``None`` if the index is
            out of bounds for the training set.
        """

        if idx >= len(self.mnist_loaders['train'].dataset):
            return None

        image, label = self.mnist_loaders['train'].dataset.__getitem__(idx)

        label_tensor = self._zeros(10)
        label_tensor[label] = 1

        return image, label_tensor

    def mnist_test_shares(self, idx):
        """
        Get the ``idx``-th testing example of the MNIST data set as additive secret shares.

        Parameters
        ----------
        idx: int
            The index of the testing example to return.

        Returns
        -------
        image_shares, label_shares:
            ``(image_shares, label_shares)``, where ``image_shares`` is a list of additive secret shares of the input (image),
            and ``label_shares`` is a list of additive secret shares of the output (one-hot label vector) of the ``idx``-th
            testing example. Returns ``None`` if the index is out of bounds for the training set.
        """

        image, label = self.mnist_loaders['test'].dataset.__getitem__(idx)

        image_shares = create_shares(image, self.num_groups)

        return image_shares, label

    def mnist_test_data(self, idx):
        """
        Get the ``idx``-th testing example of the MNIST data set as clear text tensors.

        Parameters
        ----------
        idx: int
            The index of the testing example to return.

        Returns
        -------
        image_shares, label_shares:
            ``(image_shares, label_shares)``, where ``image_shares`` is the input (image), and ``label_shares`` is
            the output (one-hot label vector) of the ``idx``-th testing example. Returns ``None`` if the index is
            out of bounds for the training set.
        """

        if idx >= len(self.mnist_loaders['test'].dataset):
            return None

        return self.mnist_loaders['test'].dataset.__getitem__(idx)

    def get_mnist_train_size(self):
        """
        Get the size of the MNIST training set.
        """

        return len(self.mnist_loaders['train'].dataset)

    def get_mnist_test_size(self):
        """
        Get the size of the MNIST testing set.
        """

        return len(self.mnist_loaders['test'].dataset)
