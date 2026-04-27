import numpy as np
from src.utils import ActivationFunctions, LossFunctions

class MultiLayerPerceptron:
    """
    Çok Katmanlı Algılayıcı (MLP)
    Karakter tanıma için gizli katmanlara sahip ağ yapısı
    """
    
    def __init__(self, layer_sizes, learning_rate=0.1, epochs=1000,
                 activation='sigmoid', loss='mse'):
        """
        Args:
            layer_sizes: Her katmandaki nöron sayısı [input, hidden1, hidden2, ..., output]
                        Örnek: [35, 20, 10] -> 35 giriş, 20 gizli, 10 çıkış
            learning_rate: Öğrenme oranı
            epochs: Eğitim iterasyonu
            activation: 'sigmoid', 'tanh', 'relu'
            loss: 'mse' veya 'crossentropy'
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_name = activation
        self.loss_name = loss
        
        # Aktivasyon fonksiyonlarını seç
        if activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_deriv = ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_deriv = ActivationFunctions.tanh_derivative
        elif activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_deriv = ActivationFunctions.relu_derivative
        else:
            self.activation = ActivationFunctions.sigmoid
            self.activation_deriv = ActivationFunctions.sigmoid_derivative
        
        # Ağırlıkları ve biasları başlat (Xavier Initialization)
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier başlatma: sqrt(2 / (n_in + n_out))
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Eğitim tarihçesi
        self.loss_history = []
        self.accuracy_history = []
        
    def _forward(self, X, store_cache=True):
        """
        İleri besleme (Forward Propagation)
        
        Args:
            X: Giriş verisi (n_samples, input_size)
            store_cache: Ara değerleri sakla (backprop için gerekli)
        
        Returns:
            output: Ağın çıkışı
            cache: Ara hesaplamalar (backprop için)
        """
        cache = {'activations': [X], 'weighted_inputs': []}
        
        current_input = X
        
        # Her katman için ileri besleme
        for i in range(len(self.weights)):
            # z = W·a + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            cache['weighted_inputs'].append(z)
            
            # a = activation(z)
            if i == len(self.weights) - 1:
                # Çıkış katmanı için sigmoid (0-1 arası)
                output = ActivationFunctions.sigmoid(z)
            else:
                # Gizli katmanlar için seçilen aktivasyon
                output = self.activation(z)
            
            cache['activations'].append(output)
            current_input = output
        
        return output, cache
    
    def _backward(self, y_true, cache):
        """
        Geri yayılım (Backpropagation)
        
        Args:
            y_true: Gerçek etiketler (n_samples, output_size)
            cache: Forward'dan gelen ara değerler
        
        Returns:
            weight_grads: Ağırlık gradyanları
            bias_grads: Bias gradyanları
        """
        n_samples = y_true.shape[0]
        weight_grads = []
        bias_grads = []
        
        # Çıkış katmanı hatası
        output = cache['activations'][-1]
        
        # MSE için gradyan: dL/doutput = (output - y_true)
        # Sigmoid türevi ile çarp
        delta = (output - y_true) * ActivationFunctions.sigmoid_derivative(output)
        
        # Katmanları geriye doğru gez
        for i in range(len(self.weights) - 1, -1, -1):
            # Önceki katmanın aktivasyonu
            prev_activation = cache['activations'][i]
            
            # Gradyanları hesapla
            dw = np.dot(prev_activation.T, delta) / n_samples
            db = np.mean(delta, axis=0, keepdims=True)
            
            weight_grads.insert(0, dw)
            bias_grads.insert(0, db)
            
            # Eğer ilk katman değilse, hatayı bir önceki katmana yay
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # Gizli katman aktivasyon türevi
                delta = delta * self.activation_deriv(cache['activations'][i])
        
        return weight_grads, bias_grads
    
    def fit(self, X, y, verbose=True, validation_data=None):
        """
        Ağı eğit
        
        Args:
            X: Eğitim verisi (n_samples, input_size)
            y: Eğitim etiketleri (n_samples, output_size) - one-hot encoded
            verbose: Eğitim bilgisi yazdır
            validation_data: Doğrulama verisi (X_val, y_val)
        """
        for epoch in range(self.epochs):
            # İleri besleme
            output, cache = self._forward(X)
            
            # Geri yayılım
            weight_grads, bias_grads = self._backward(y, cache)
            
            # Ağırlıkları güncelle (Gradient Descent)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_grads[i]
                self.biases[i] -= self.learning_rate * bias_grads[i]
            
            # Hata hesapla
            loss = LossFunctions.mean_squared_error(y, output)
            self.loss_history.append(loss)
            
            # Doğruluk hesapla
            accuracy = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
            self.accuracy_history.append(accuracy)
            
            # Doğrulama verisi varsa
            val_loss = None
            val_acc = None
            if validation_data is not None:
                X_val, y_val = validation_data
                output_val, _ = self._forward(X_val, store_cache=False)
                val_loss = LossFunctions.mean_squared_error(y_val, output_val)
                val_acc = np.mean(np.argmax(output_val, axis=1) == np.argmax(y_val, axis=1))
            
            # İlerleme yazdır
            if verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Loss: {loss:.6f} - Acc: {accuracy:.4f} - "
                          f"Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Loss: {loss:.6f} - Acc: {accuracy:.4f}")
    
    def predict(self, X):
        """
        Tahmin yap
        
        Args:
            X: Giriş verisi (n_samples, input_size) veya (input_size,)
        
        Returns:
            predictions: (n_samples, output_size) şeklinde olasılıklar
            predicted_classes: (n_samples,) şeklinde tahmin edilen sınıflar
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        output, _ = self._forward(X, store_cache=False)
        predicted_classes = np.argmax(output, axis=1)
        
        return output, predicted_classes
    
    def predict_single(self, X):
        """Tek örnek için tahmin"""
        output, predicted_class = self.predict(X)
        return predicted_class[0], output[0]
    
    def save_weights(self, filepath):
        """Ağırlıkları kaydet"""
        np.savez(filepath, 
                 weights=self.weights,
                 biases=self.biases,
                 layer_sizes=self.layer_sizes)
        print(f"Ağırlıklar {filepath} dosyasına kaydedildi")
    
    def load_weights(self, filepath):
        """Ağırlıkları yükle"""
        data = np.load(filepath, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
        self.layer_sizes = data['layer_sizes']
        print(f"Ağırlıklar {filepath} dosyasından yüklendi")


print("mlp.py başarıyla yüklendi")