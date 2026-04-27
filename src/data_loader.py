import numpy as np
import os


class CharacterDataLoader:
    """
    Karakter tanıma için veri yükleme ve oluşturma sınıfı.
    Karakterler 7x5 binary matris olarak temsil edilir.
    """

    def __init__(self, matrix_size=(7, 5)):
        """
        Args:
            matrix_size: Karakter matris boyutu. Varsayılan: 7 satır, 5 sütun.
        """

        self.matrix_size = matrix_size
        self.input_size = matrix_size[0] * matrix_size[1]

    def create_sample_data(self):
        """
        Örnek karakter verisi oluşturur.
        Harfler: A, B, C, D, E
        Fontlar: standart, italik, kalın
        Matris boyutu: 7x5
        """

        characters = {
            'A': {
                'standart': np.array([
                    [0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1]
                ]).flatten(),

                'italik': np.array([
                    [0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0]
                ]).flatten(),

                'kalin': np.array([
                    [0, 1, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1]
                ]).flatten()
            },

            'B': {
                'standart': np.array([
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 0]
                ]).flatten(),

                'italik': np.array([
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 0]
                ]).flatten(),

                'kalin': np.array([
                    [1, 1, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0]
                ]).flatten()
            },

            'C': {
                'standart': np.array([
                    [0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 1, 1, 0]
                ]).flatten(),

                'italik': np.array([
                    [0, 0, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1]
                ]).flatten(),

                'kalin': np.array([
                    [0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1]
                ]).flatten()
            },

            'D': {
                'standart': np.array([
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 0]
                ]).flatten(),

                'italik': np.array([
                    [0, 1, 1, 1, 0],
                    [0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 1, 1, 0, 0]
                ]).flatten(),

                'kalin': np.array([
                    [1, 1, 1, 1, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0]
                ]).flatten()
            },

            'E': {
                'standart': np.array([
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]
                ]).flatten(),

                'italik': np.array([
                    [0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]
                ]).flatten(),

                'kalin': np.array([
                    [1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1]
                ]).flatten()
            }
        }

        return characters

    def create_training_data(self, characters_dict):
        """
        Karakter sözlüğünden eğitim verisi oluşturur.

        Args:
            characters_dict:
                {
                    'A': {
                        'standart': array,
                        'italik': array,
                        'kalin': array
                    },
                    ...
                }

        Returns:
            X: Giriş matrisi
            y_one_hot: One-hot encoded etiket matrisi
            class_names: Sınıf isimleri
        """

        X_list = []
        y_list = []

        class_names = sorted(list(characters_dict.keys()))
        class_to_index = {class_name: i for i, class_name in enumerate(class_names)}

        for char_name, fonts in characters_dict.items():
            for font_name, matrix in fonts.items():
                X_list.append(matrix)
                y_list.append(class_to_index[char_name])

        X = np.array(X_list)
        y = np.array(y_list)

        y_one_hot = np.zeros((len(y), len(class_names)))

        for i, label in enumerate(y):
            y_one_hot[i, label] = 1

        print(f"✓ Eğitim verisi oluşturuldu: {X.shape[0]} örnek, {len(class_names)} sınıf")
        print(f"  Sınıflar: {class_names}")

        return X, y_one_hot, class_names

    def load_from_file(self, filepath):
        """
        .txt veya .csv dosyasından karakter verisi yükler.
        Format:
            feature1,feature2,...,label
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

        data = []
        labels = []

        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "," in line:
                    values = line.split(",")
                else:
                    values = line.split()

                features = [float(value) for value in values[:-1]]
                label = values[-1].strip()

                data.append(features)
                labels.append(label)

        X = np.array(data)
        y = np.array(labels)

        print(f"✓ {X.shape[0]} örnek yüklendi: {filepath}")

        return X, y, labels

    def save_to_file(self, X, y, filepath):
        """
        Veriyi dosyaya kaydeder.
        """

        with open(filepath, "w", encoding="utf-8") as file:
            file.write("# Format: feature1,feature2,...,label\n")

            for i in range(X.shape[0]):
                features = ",".join(map(str, X[i]))
                file.write(f"{features},{y[i]}\n")

        print(f"✓ Veri kaydedildi: {filepath}")

    def visualize_character(self, matrix, title="Karakter"):
        """
        Tek bir karakter matrisini matplotlib ile gösterir.
        """

        import matplotlib.pyplot as plt

        matrix = np.array(matrix)

        if matrix.ndim == 1:
            matrix = matrix.reshape(self.matrix_size)

        plt.figure(figsize=(3, 4))
        plt.imshow(matrix, cmap="binary", interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        plt.show()


class LogicGateDataLoader:
    """
    AND, OR ve XOR kapıları için doğruluk tablosu üretir.
    """

    @staticmethod
    def get_AND():
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        y = np.array([
            [0],
            [0],
            [0],
            [1]
        ])

        return X, y

    @staticmethod
    def get_OR():
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        y = np.array([
            [0],
            [1],
            [1],
            [1]
        ])

        return X, y

    @staticmethod
    def get_XOR():
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

        y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

        return X, y

    @staticmethod
    def get_gate(gate_name):
        gate_name = gate_name.upper()

        if gate_name == "AND":
            return LogicGateDataLoader.get_AND()

        elif gate_name == "OR":
            return LogicGateDataLoader.get_OR()

        elif gate_name == "XOR":
            return LogicGateDataLoader.get_XOR()

        else:
            raise ValueError(f"Bilinmeyen lojik kapı: {gate_name}")


print("data_loader.py başarıyla yüklendi")