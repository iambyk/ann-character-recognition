import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.random.seed(42)
import src.perceptron
from src.mlp import MultiLayerPerceptron
import src.data_loader

def test_perceptron():
    """Perceptron'u AND, OR, XOR ile test et"""
    print("\n" + "="*50)
    print("PERCEPTRON TESTİ")
    print("="*50)
    
    gates = ['AND', 'OR', 'XOR']
    
    for gate in gates:
        print(f"\n--- {gate} KAPISI ---")
        X, y = src.data_loader.LogicGateDataLoader.get_gate(gate)
        
        # Perceptron oluştur ve eğit
        p = src.perceptron.Perceptron(input_size=2, learning_rate=0.1, epochs=1000)
        
        # ÖNEMLİ: y'yi düzleştiriyoruz (flatten)
        p.fit(X, y.flatten(), verbose=False)
        
        # Tahminleri kontrol et
        predictions = p.predict(X)
        predictions_binary = (predictions > 0.5).astype(int)
        
        print(f"Girdi:\n{X}")
        print(f"Beklenen: {y.flatten().tolist()}")
        print(f"Tahmin: {predictions_binary.tolist()}")
        
        accuracy = np.mean(predictions_binary == y.flatten())
        print(f"Doğruluk: {accuracy*100:.1f}%")

def test_mlp():
    """MLP'yi XOR ve karakter verisi ile test et"""
    print("\n" + "="*50)
    print("MLP TESTİ")
    print("="*50)
    
    # XOR testi (Perceptron çözemez, MLP çözer)
    print("\n--- XOR PROBLEMI (MLP ile) ---")
    X, y = src.data_loader.LogicGateDataLoader.get_XOR()
    
    mlp = MultiLayerPerceptron(
        layer_sizes=[2, 4, 1],
        learning_rate=0.5,
        epochs=5000
    )
    mlp.fit(X, y, verbose=False)
    
    output, predicted = mlp.predict(X)
    predictions_binary = (output > 0.5).astype(int)
    
    print(f"Girdi: {X.tolist()}")
    print(f"Beklenen: {y.flatten().tolist()}")
    print(f"Tahmin: {predictions_binary.flatten().tolist()}")
    
    accuracy = np.mean(predictions_binary.flatten() == y.flatten())
    print(f"Doğruluk: {accuracy*100:.1f}%")
    
    # Karakter tanıma testi
    print("\n--- KARAKTER TANIMA (Örnek Veri) ---")
    loader = src.data_loader.CharacterDataLoader(matrix_size=(5, 7))
    characters = loader.create_sample_data()
    
    X, y, class_names = loader.create_training_data(characters)
    
    mlp_char = MultiLayerPerceptron(
    layer_sizes=[X.shape[1], 25, len(class_names)],
    learning_rate=0.3,
    epochs=2000
)
    mlp_char.fit(X, y, verbose=True)
    
    # Test et
    output, predicted = mlp_char.predict(X)
    accuracy = np.mean(predicted == np.argmax(y, axis=1))
    print(f"\nEğitim Doğruluğu: {accuracy*100:.1f}%")


if __name__ == "__main__":
    test_perceptron()
    test_mlp()
    print("\n" + "="*50)
    print("TESTLER TAMAMLANDI")
    print("="*50)