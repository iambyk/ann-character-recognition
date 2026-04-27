import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.perceptron import Perceptron
from src.mlp import MultiLayerPerceptron
from src.data_loader import LogicGateDataLoader, CharacterDataLoader
from src.visualizer import Visualizer


np.random.seed(42)


def test_logic_gate_visualization():
    gates = ["AND", "OR", "XOR"]

    for gate in gates:
        X, y = LogicGateDataLoader.get_gate(gate)

        perceptron = Perceptron(
            input_size=2,
            learning_rate=0.1,
            epochs=100
        )

        perceptron.fit(X, y.flatten(), verbose=False)

        predictions = perceptron.predict(X)
        predictions_binary = (predictions > 0.5).astype(int)

        accuracy = np.mean(predictions_binary == y.flatten())
        print(f"{gate} kapısı doğruluk: {accuracy * 100:.1f}%")

        Visualizer.plot_logic_gate_decision_boundary(
            perceptron=perceptron,
            X=X,
            y=y,
            gate_name=gate
        )


def test_character_visualization():
    loader = CharacterDataLoader(matrix_size=(7, 5))
    characters = loader.create_sample_data()

    X, y, class_names = loader.create_training_data(characters)

    labels = []
    for char_name, fonts in characters.items():
        for font_name in fonts.keys():
            labels.append(f"{char_name}-{font_name}")

    Visualizer.plot_character_dataset(
        X=X,
        labels=labels,
        matrix_size=(7, 5),
        title="5 Harf - 3 Font Karakter Veri Seti"
    )

    mlp = MultiLayerPerceptron(
        layer_sizes=[X.shape[1], 25, len(class_names)],
        learning_rate=0.3,
        epochs=2000
    )

    mlp.fit(X, y, verbose=True)

    Visualizer.plot_training_history(
        loss_history=mlp.loss_history,
        accuracy_history=mlp.accuracy_history,
        title="MLP Karakter Tanıma Eğitim Grafiği"
    )


if __name__ == "__main__":
    test_logic_gate_visualization()
    test_character_visualization()

    print("\nGörselleştirme testleri tamamlandı.")