import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """
    Eğitim süreci, karakter matrisleri ve karar sınırları için görselleştirme sınıfı.
    """

    @staticmethod
    def plot_training_history(loss_history, accuracy_history=None, title="Eğitim Grafiği"):
        """
        Eğitim sürecindeki loss ve accuracy değerlerini çizer.
        """

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label="Loss", color="red", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if accuracy_history is not None:
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax2.plot(accuracy_history, label="Accuracy", color="blue", linewidth=2)
            ax2.set_ylabel("Accuracy")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
        else:
            plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_logic_gate_decision_boundary(perceptron, X, y, gate_name="AND"):
        """
        Perceptron karar sınırını 2 boyutlu uzayda çizer.
        """

        y = np.array(y).flatten()

        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5

        xx, yy, Z = perceptron.get_decision_boundary(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=300
        )

        plt.figure(figsize=(7, 6))

        plt.contourf(
            xx,
            yy,
            Z,
            levels=[-0.1, 0.5, 1.1],
            alpha=0.3,
            cmap="coolwarm"
        )

        for class_value in np.unique(y):
            indices = y == class_value
            plt.scatter(
                X[indices, 0],
                X[indices, 1],
                label=f"Sınıf {int(class_value)}",
                s=120,
                edgecolors="black"
            )

        if hasattr(perceptron, "weights"):
            w1 = perceptron.weights[0]
            w2 = perceptron.weights[1]
            b = perceptron.weights[2]

            if abs(w2) > 1e-8:
                x_values = np.linspace(x_min, x_max, 100)
                y_values = -(w1 * x_values + b) / w2

                plt.plot(
                    x_values,
                    y_values,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label="Karar Sınırı"
                )

        plt.title(f"{gate_name} Kapısı - Perceptron Karar Sınırı")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_character(matrix, matrix_size=(7, 5), title="Karakter"):
        """
        Tek bir karakter matrisini çizer.
        """

        matrix = np.array(matrix)

        if matrix.ndim == 1:
            matrix = matrix.reshape(matrix_size)

        plt.figure(figsize=(3, 4))
        plt.imshow(matrix, cmap="binary", interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_character_dataset(X, labels=None, matrix_size=(7, 5), title="Karakter Veri Seti"):
        """
        Birden fazla karakter örneğini grid halinde çizer.
        """

        n_samples = X.shape[0]
        cols = 5
        rows = int(np.ceil(n_samples / cols))

        plt.figure(figsize=(cols * 2, rows * 2.5))

        for i in range(n_samples):
            plt.subplot(rows, cols, i + 1)

            character_matrix = X[i].reshape(matrix_size)
            plt.imshow(character_matrix, cmap="binary", interpolation="nearest")

            if labels is not None:
                plt.title(str(labels[i]))
            else:
                plt.title(f"Örnek {i + 1}")

            plt.axis("off")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


print("visualizer.py başarıyla yüklendi")