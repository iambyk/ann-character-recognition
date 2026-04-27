import numpy as np

class ActivationFunctions:
    """Aktivasyon fonksiyonları ve türevleri"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid aktivasyon fonksiyonu"""
        # Sayısal stabilite için clip işlemi
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(output):
        """Sigmoid'in türevi (çıkış değeri kullanarak)"""
        return output * (1 - output)
    
    @staticmethod
    def relu(x):
        """ReLU aktivasyon fonksiyonu"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU'nun türevi"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh(x):
        """Tanh aktivasyon fonksiyonu"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(output):
        """Tanh'ın türevi"""
        return 1 - output**2
    
    @staticmethod
    def step(x, threshold=0):
        """Step/Heaviside fonksiyonu (Perceptron için)"""
        return np.where(x >= threshold, 1, 0)


class LossFunctions:
    """Hata/Kayıp fonksiyonları"""
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """MSE: Mean Squared Error"""
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary Crossentropy (0-1 arası değerler için)"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))


def normalize(X, method='minmax'):
    """
    Veriyi normalize etme
    
    Args:
        X: Giriş verisi (n_samples, n_features)
        method: 'minmax' veya 'zscore'
    """
    if method == 'minmax':
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        return (X - X_min) / (X_max - X_min + 1e-8), (X_min, X_max)
    
    elif method == 'zscore':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / (std + 1e-8), (mean, std)


def denormalize(X, params, method='minmax'):
    """Normalizasyonu geri alma"""
    if method == 'minmax':
        X_min, X_max = params
        return X * (X_max - X_min) + X_min
    
    elif method == 'zscore':
        mean, std = params
        return X * std + mean


def one_hot_encode(y, num_classes):
    """
    Sınıf etiketlerini one-hot encoding'e çevirme
    
    Args:
        y: (n_samples,) şeklinde etiket vektörü
        num_classes: Toplam sınıf sayısı
    
    Returns:
        (n_samples, num_classes) şeklinde one-hot matris
    """
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, int(label)] = 1
    return one_hot


def one_hot_decode(y_one_hot):
    """One-hot encoding'den sınıf etiketlerine dönüştürme"""
    return np.argmax(y_one_hot, axis=1)


print("utils.py başarıyla yüklendi")