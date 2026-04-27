import numpy as np
from src.utils import ActivationFunctions


class Perceptron:
    """
    Tek katmanlı Perceptron
    Lojik kapılar için kullanılacaktır: AND, OR, XOR
    """

    def __init__(self, input_size, learning_rate=0.1, epochs=100, activation='step'):
        """
        Args:
            input_size: Giriş sayısı
            learning_rate: Öğrenme oranı
            epochs: Eğitim epoch sayısı
            activation: Aktivasyon fonksiyonu: 'step', 'sigmoid', 'tanh'
        """

        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation

        # input_size kadar ağırlık + 1 bias ağırlığı
        self.weights = np.random.uniform(-1, 1, input_size + 1)

        self.loss_history = []
        self.training_data = None
        self.training_labels = None

    def _forward(self, X):
        """
        İleri besleme işlemi

        Args:
            X: Giriş matrisi, shape: (n_samples, input_size)

        Returns:
            raw_output: Net giriş değeri
            output: Aktivasyon sonrası çıktı
        """

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Bias için X'e 1'lerden oluşan sütun eklenir
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])

        # z = Xw + b
        raw_output = np.dot(X_with_bias, self.weights)

        if self.activation == 'step':
            output = ActivationFunctions.step(raw_output)
        elif self.activation == 'sigmoid':
            output = ActivationFunctions.sigmoid(raw_output)
        elif self.activation == 'tanh':
            output = ActivationFunctions.tanh(raw_output)
        else:
            output = raw_output

        return raw_output, output

    def fit(self, X, y, verbose=True):
        """
        Perceptron'u eğit

        Args:
            X: Eğitim verisi, shape: (n_samples, input_size)
            y: Etiketler, shape: (n_samples,)
            verbose: Eğitim sırasında çıktı yazdırılsın mı?
        """

        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if y.ndim == 2:
            y = y.flatten()

        self.training_data = X.copy()
        self.training_labels = y.copy()

        for epoch in range(self.epochs):
            epoch_error = 0

            for i in range(X.shape[0]):
                sample = X[i:i + 1]
                target = y[i]

                _, prediction = self._forward(sample)
                prediction = prediction[0]

                error = target - prediction
                epoch_error += error ** 2

                # Bias dahil giriş vektörü
                sample_with_bias = np.append(sample.flatten(), 1)

                # Perceptron ağırlık güncellemesi
                self.weights += self.learning_rate * error * sample_with_bias

            mse = epoch_error / X.shape[0]
            self.loss_history.append(mse)

            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - MSE: {mse:.6f}")

    def predict(self, X):
        """
        Tahmin yap

        Args:
            X: Test verisi

        Returns:
            output: Tahmin sonuçları
        """

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        _, output = self._forward(X)
        return output

    def get_decision_boundary(self, x_min, x_max, y_min, y_max, resolution=100):
        """
        2 boyutlu lojik kapı problemi için karar sınırı üretir.
        GUI kısmında matplotlib ile çizdirmek için kullanılacak.
        """

        x_values = np.linspace(x_min, x_max, resolution)
        y_values = np.linspace(y_min, y_max, resolution)

        xx, yy = np.meshgrid(x_values, y_values)

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        return xx, yy, Z


print("perceptron.py başarıyla yüklendi")