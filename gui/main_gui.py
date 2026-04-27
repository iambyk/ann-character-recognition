import customtkinter as ctk
import numpy as np
from tkinter import messagebox, filedialog

from src.mlp import MultiLayerPerceptron
from src.perceptron import Perceptron
from src.data_loader import CharacterDataLoader, LogicGateDataLoader
from src.visualizer import Visualizer


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ANNProjectGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Yapay Sinir Ağları Projesi - Karakter Tanıma ve Lojik Kapılar")
        self.geometry("1250x780")
        self.minsize(1100, 700)

        self.matrix_size = (7, 5)
        self.draw_matrix = np.zeros(self.matrix_size, dtype=int)
        self.grid_buttons = []

        self.mlp_model = None
        self.class_names = []
        self.character_X = None
        self.character_y = None
        self.characters_dict = None

        self.perceptron_model = None
        self.logic_X = None
        self.logic_y = None
        self.selected_gate_name = None

        self.create_widgets()

    def create_widgets(self):
        ctk.CTkLabel(
            self,
            text="Yapay Sinir Ağları Projesi",
            font=("Arial", 24, "bold")
        ).pack(pady=(12, 2))

        ctk.CTkLabel(
            self,
            text="MLP ile Karakter Tanıma  |  Perceptron ile Lojik Kapı Sınıflandırma",
            font=("Arial", 13)
        ).pack(pady=(0, 8))

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=15, pady=5, fill="both", expand=True)

        self.character_tab = self.tabview.add("Karakter Tanıma")
        self.logic_tab = self.tabview.add("Lojik Kapılar")

        self.create_character_tab()
        self.create_logic_tab()

    # ============================================================
    # KARAKTER TANIMA
    # ============================================================

    def create_character_tab(self):
        left_panel = ctk.CTkFrame(self.character_tab, width=280)
        left_panel.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left_panel.pack_propagate(False)

        right_panel = ctk.CTkScrollableFrame(self.character_tab)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        # Sol panel
        ctk.CTkLabel(
            left_panel,
            text="MLP Parametreleri",
            font=("Arial", 16, "bold")
        ).pack(pady=(15, 10))

        ctk.CTkLabel(left_panel, text="Learning Rate:").pack(anchor="w", padx=20)
        self.char_lr_entry = ctk.CTkEntry(left_panel, width=200)
        self.char_lr_entry.insert(0, "0.3")
        self.char_lr_entry.pack(pady=(3, 8))

        ctk.CTkLabel(left_panel, text="Epoch Sayısı:").pack(anchor="w", padx=20)
        self.char_epoch_entry = ctk.CTkEntry(left_panel, width=200)
        self.char_epoch_entry.insert(0, "2000")
        self.char_epoch_entry.pack(pady=(3, 8))

        ctk.CTkLabel(left_panel, text="Gizli Katman Nöron Sayısı:").pack(anchor="w", padx=20)
        self.char_hidden_entry = ctk.CTkEntry(left_panel, width=200)
        self.char_hidden_entry.insert(0, "25")
        self.char_hidden_entry.pack(pady=(3, 8))

        ctk.CTkButton(
            left_panel,
            text="MLP Modelini Eğit",
            width=200,
            command=self.train_character_model
        ).pack(pady=6)

        ctk.CTkButton(
            left_panel,
            text="Karakter Veri Setini Göster",
            width=200,
            command=self.show_character_dataset
        ).pack(pady=6)

        self.plot_char_button = ctk.CTkButton(
            left_panel,
            text="Eğitim Grafiğini Göster",
            width=200,
            command=self.plot_character_training_history,
            state="disabled"
        )
        self.plot_char_button.pack(pady=6)

        ctk.CTkLabel(left_panel, text="────────────────────", text_color="#555555").pack(pady=8)

        ctk.CTkLabel(
            left_panel,
            text="Model Yönetimi",
            font=("Arial", 13, "bold")
        ).pack(pady=3)

        self.save_model_button = ctk.CTkButton(
            left_panel,
            text="Modeli Kaydet",
            width=200,
            command=self.save_mlp_model,
            state="disabled"
        )
        self.save_model_button.pack(pady=6)

        ctk.CTkButton(
            left_panel,
            text="Modeli Yükle",
            width=200,
            command=self.load_mlp_model
        ).pack(pady=6)

        ctk.CTkLabel(left_panel, text="────────────────────", text_color="#555555").pack(pady=8)

        self.char_status_label = ctk.CTkLabel(
            left_panel,
            text="Durum: Henüz eğitilmedi.",
            wraplength=240,
            justify="left"
        )
        self.char_status_label.pack(pady=5, padx=10)

        self.char_classes_label = ctk.CTkLabel(
            left_panel,
            text="Sınıflar: -",
            wraplength=240,
            justify="left"
        )
        self.char_classes_label.pack(pady=5, padx=10)

        # Sağ panel
        ctk.CTkLabel(
            right_panel,
            text="7×5 Karakter Çizim Alanı",
            font=("Arial", 18, "bold")
        ).pack(pady=(15, 3))

        ctk.CTkLabel(
            right_panel,
            text="Hücrelere tıklayarak piksel aç/kapat. Sonra 'Tahmin Et' butonuna bas.",
            font=("Arial", 12)
        ).pack(pady=(0, 10))

        self.draw_grid_frame = ctk.CTkFrame(right_panel)
        self.draw_grid_frame.pack(pady=5)

        self.create_drawing_grid()

        button_row = ctk.CTkFrame(right_panel, fg_color="transparent")
        button_row.pack(pady=12)

        ctk.CTkButton(
            button_row,
            text="Tahmin Et",
            width=140,
            command=self.predict_drawn_character
        ).grid(row=0, column=0, padx=8)

        ctk.CTkButton(
            button_row,
            text="Temizle",
            width=140,
            fg_color="#8a1f1f",
            hover_color="#a83232",
            command=self.clear_drawing_grid
        ).grid(row=0, column=1, padx=8)

        ctk.CTkButton(
            button_row,
            text="Çizimi Kaydet",
            width=140,
            fg_color="#1f5a8a",
            hover_color="#2a6ba8",
            command=self.save_drawn_character
        ).grid(row=0, column=2, padx=8)

        self.prediction_result_label = ctk.CTkLabel(
            right_panel,
            text="Tahmin Sonucu: -",
            font=("Arial", 24, "bold")
        )
        self.prediction_result_label.pack(pady=(15, 4))

        self.probability_label = ctk.CTkLabel(
            right_panel,
            text="Olasılıklar: -",
            wraplength=650,
            justify="left"
        )
        self.probability_label.pack(pady=4)

        ctk.CTkLabel(right_panel, text="").pack(pady=20)

    def create_drawing_grid(self):
        self.grid_buttons = []

        for r in range(self.matrix_size[0]):
            row_buttons = []
            for c in range(self.matrix_size[1]):
                btn = ctk.CTkButton(
                    self.draw_grid_frame,
                    text="",
                    width=60,
                    height=55,
                    fg_color="#FFFFFF",
                    hover_color="#CCCCCC",
                    border_width=1,
                    border_color="#555555",
                    command=lambda row=r, col=c: self.toggle_grid_cell(row, col)
                )
                btn.grid(row=r, column=c, padx=3, pady=3)
                row_buttons.append(btn)

            self.grid_buttons.append(row_buttons)

    def toggle_grid_cell(self, row, col):
        self.draw_matrix[row, col] = 1 - self.draw_matrix[row, col]
        self.update_drawing_grid_colors()

    def update_drawing_grid_colors(self):
        for r in range(self.matrix_size[0]):
            for c in range(self.matrix_size[1]):
                if self.draw_matrix[r, c] == 1:
                    self.grid_buttons[r][c].configure(fg_color="#111111", hover_color="#333333")
                else:
                    self.grid_buttons[r][c].configure(fg_color="#FFFFFF", hover_color="#CCCCCC")

    def clear_drawing_grid(self):
        self.draw_matrix = np.zeros(self.matrix_size, dtype=int)
        self.update_drawing_grid_colors()
        self.prediction_result_label.configure(text="Tahmin Sonucu: -")
        self.probability_label.configure(text="Olasılıklar: -")

    def train_character_model(self):
        try:
            lr = float(self.char_lr_entry.get())
            epochs = int(self.char_epoch_entry.get())
            hidden = int(self.char_hidden_entry.get())

            if lr <= 0 or epochs <= 0 or hidden <= 0:
                raise ValueError("Tüm değerler pozitif olmalıdır.")

        except ValueError as e:
            messagebox.showerror("Parametre Hatası", str(e))
            return

        try:
            self.char_status_label.configure(text="Durum: Eğitiliyor...")
            self.update_idletasks()

            np.random.seed(42)

            loader = CharacterDataLoader(matrix_size=self.matrix_size)
            self.characters_dict = loader.create_sample_data()
            X, y, class_names = loader.create_training_data(self.characters_dict)

            self.character_X = X
            self.character_y = y
            self.class_names = class_names

            self.mlp_model = MultiLayerPerceptron(
                layer_sizes=[X.shape[1], hidden, len(class_names)],
                learning_rate=lr,
                epochs=epochs
            )
            self.mlp_model.fit(X, y, verbose=False)

            output, predicted = self.mlp_model.predict(X)
            accuracy = np.mean(predicted == np.argmax(y, axis=1))

            self.char_status_label.configure(
                text=f"Durum: Tamamlandı.\nEğitim Doğruluğu: %{accuracy * 100:.2f}"
            )
            self.char_classes_label.configure(text=f"Sınıflar: {', '.join(self.class_names)}")
            self.plot_char_button.configure(state="normal")
            self.save_model_button.configure(state="normal")

            messagebox.showinfo(
                "Eğitim Tamamlandı",
                f"Model başarıyla eğitildi.\nEğitim Doğruluğu: %{accuracy * 100:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Hata", str(e))
            self.char_status_label.configure(text="Durum: Hata oluştu.")

    def predict_drawn_character(self):
        if self.mlp_model is None:
            messagebox.showwarning("Model Yok", "Önce modeli eğitmelisiniz.")
            return

        input_vector = self.draw_matrix.flatten().reshape(1, -1)
        output, predicted = self.mlp_model.predict(input_vector)

        idx = predicted[0]
        label = self.class_names[idx]
        confidence = output[0][idx] * 100

        self.prediction_result_label.configure(
            text=f"Tahmin Sonucu: {label}   (%{confidence:.1f})"
        )

        prob_text = "Olasılıklar:\n"
        for name, prob in zip(self.class_names, output[0]):
            prob_text += f"{name}: %{prob * 100:.1f}   "

        self.probability_label.configure(text=prob_text)

    def show_character_dataset(self):
        try:
            loader = CharacterDataLoader(matrix_size=self.matrix_size)
            characters = loader.create_sample_data()
            X, y, class_names = loader.create_training_data(characters)

            labels = []
            for char_name, fonts in characters.items():
                for font_name in fonts.keys():
                    labels.append(f"{char_name}-{font_name}")

            Visualizer.plot_character_dataset(
                X=X,
                labels=labels,
                matrix_size=self.matrix_size,
                title="5 Harf - 3 Font Karakter Veri Seti"
            )
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def plot_character_training_history(self):
        if self.mlp_model is None:
            messagebox.showwarning("Model Yok", "Önce modeli eğitmelisiniz.")
            return

        Visualizer.plot_training_history(
            loss_history=self.mlp_model.loss_history,
            accuracy_history=self.mlp_model.accuracy_history,
            title="MLP Karakter Tanıma Eğitim Grafiği"
        )

    def save_mlp_model(self):
        if self.mlp_model is None:
            messagebox.showwarning("Model Yok", "Kaydedilecek model yok.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Dosyası", "*.npz")],
            initialfile="mlp_model.npz"
        )

        if filepath:
            try:
                self.mlp_model.save_weights(filepath)
                messagebox.showinfo("Başarılı", f"Model kaydedildi:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    def load_mlp_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("NumPy Dosyası", "*.npz")]
        )

        if filepath:
            try:
                self.mlp_model = MultiLayerPerceptron(
                    layer_sizes=[35, 25, 5],
                    learning_rate=0.1,
                    epochs=1
                )
                self.mlp_model.load_weights(filepath)
                self.class_names = ['A', 'B', 'C', 'D', 'E']

                self.char_status_label.configure(text="Durum: Model yüklendi.")
                self.char_classes_label.configure(text=f"Sınıflar: {', '.join(self.class_names)}")
                self.plot_char_button.configure(state="normal")
                self.save_model_button.configure(state="normal")

                messagebox.showinfo("Başarılı", "Model başarıyla yüklendi.")
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    def save_drawn_character(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Dosyası", "*.csv")],
            initialfile="cizilen_karakter.csv"
        )

        if filepath:
            try:
                flat = self.draw_matrix.flatten()
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("# 7x5 karakter matrisi (düzleştirilmiş)\n")
                    f.write(",".join(map(str, flat)) + "\n")
                messagebox.showinfo("Başarılı", f"Çizim kaydedildi:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    # ============================================================
    # LOJİK KAPILAR
    # ============================================================

    def create_logic_tab(self):
        left_panel = ctk.CTkFrame(self.logic_tab, width=280)
        left_panel.pack(side="left", fill="y", padx=(10, 5), pady=10)
        left_panel.pack_propagate(False)

        right_panel = ctk.CTkFrame(self.logic_tab)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        ctk.CTkLabel(
            left_panel,
            text="Perceptron Parametreleri",
            font=("Arial", 16, "bold")
        ).pack(pady=(15, 10))

        ctk.CTkLabel(left_panel, text="Lojik Kapı Seçimi:").pack(anchor="w", padx=20)

        self.gate_var = ctk.StringVar(value="AND")
        ctk.CTkOptionMenu(
            left_panel,
            values=["AND", "OR", "XOR"],
            variable=self.gate_var,
            width=200
        ).pack(pady=(3, 8))

        ctk.CTkLabel(left_panel, text="Learning Rate:").pack(anchor="w", padx=20)
        self.logic_lr_entry = ctk.CTkEntry(left_panel, width=200)
        self.logic_lr_entry.insert(0, "0.1")
        self.logic_lr_entry.pack(pady=(3, 8))

        ctk.CTkLabel(left_panel, text="Epoch Sayısı:").pack(anchor="w", padx=20)
        self.logic_epoch_entry = ctk.CTkEntry(left_panel, width=200)
        self.logic_epoch_entry.insert(0, "100")
        self.logic_epoch_entry.pack(pady=(3, 8))

        ctk.CTkButton(
            left_panel,
            text="Perceptron Eğit",
            width=200,
            command=self.train_logic_gate_model
        ).pack(pady=8)

        self.plot_logic_button = ctk.CTkButton(
            left_panel,
            text="Karar Sınırını Göster",
            width=200,
            command=self.plot_logic_decision_boundary,
            state="disabled"
        )
        self.plot_logic_button.pack(pady=6)

        ctk.CTkLabel(left_panel, text="────────────────────", text_color="#555555").pack(pady=8)

        self.logic_status_label = ctk.CTkLabel(
            left_panel,
            text="Durum: Henüz eğitilmedi.",
            wraplength=240,
            justify="left"
        )
        self.logic_status_label.pack(pady=5, padx=10)

        ctk.CTkLabel(
            right_panel,
            text="Lojik Kapı Sonuçları",
            font=("Arial", 18, "bold")
        ).pack(pady=(15, 10))

        self.logic_output_textbox = ctk.CTkTextbox(
            right_panel,
            font=("Consolas", 14)
        )
        self.logic_output_textbox.pack(padx=15, pady=(0, 15), fill="both", expand=True)

        self.logic_output_textbox.insert(
            "end",
            "Burada Perceptron eğitim sonuçları gösterilecektir.\n\n"
            "1) Sol panelden AND / OR / XOR seçin.\n"
            "2) Learning rate ve epoch değerini girin.\n"
            "3) 'Perceptron Eğit' butonuna basın.\n"
            "4) Ardından karar sınırını görselleştirin.\n"
        )
        self.logic_output_textbox.configure(state="disabled")

    def train_logic_gate_model(self):
        try:
            gate_name = self.gate_var.get()
            lr = float(self.logic_lr_entry.get())
            epochs = int(self.logic_epoch_entry.get())

            if lr <= 0 or epochs <= 0:
                raise ValueError("Değerler pozitif olmalıdır.")

        except ValueError as e:
            messagebox.showerror("Parametre Hatası", str(e))
            return

        try:
            np.random.seed(42)

            X, y = LogicGateDataLoader.get_gate(gate_name)
            perceptron = Perceptron(input_size=2, learning_rate=lr, epochs=epochs)
            perceptron.fit(X, y.flatten(), verbose=False)

            predictions = perceptron.predict(X)
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = np.mean(predictions_binary == y.flatten())

            self.perceptron_model = perceptron
            self.logic_X = X
            self.logic_y = y
            self.selected_gate_name = gate_name

            self.logic_status_label.configure(
                text=f"Durum: {gate_name} eğitildi.\nDoğruluk: %{accuracy * 100:.2f}"
            )
            self.plot_logic_button.configure(state="normal")

            result = ""
            result += f"{gate_name} KAPISI - PERCEPTRON SONUÇLARI\n"
            result += "=" * 45 + "\n\n"
            result += f"Learning Rate  : {lr}\n"
            result += f"Epoch          : {epochs}\n"
            result += f"Doğruluk       : %{accuracy * 100:.2f}\n\n"

            result += "┌──────┬──────┬──────────┬──────────┐\n"
            result += "│  x1  │  x2  │ Beklenen │  Tahmin  │\n"
            result += "├──────┼──────┼──────────┼──────────┤\n"

            for i in range(X.shape[0]):
                x1 = int(X[i, 0])
                x2 = int(X[i, 1])
                exp = int(y.flatten()[i])
                pred = int(predictions_binary[i])
                check = "✓" if exp == pred else "✗"
                result += f"│  {x1}   │  {x2}   │    {exp}     │  {pred}  {check}   │\n"

            result += "└──────┴──────┴──────────┴──────────┘\n\n"
            result += f"w1   : {perceptron.weights[0]:.4f}\n"
            result += f"w2   : {perceptron.weights[1]:.4f}\n"
            result += f"bias : {perceptron.weights[2]:.4f}\n"

            if gate_name == "XOR":
                result += "\n" + "─" * 45 + "\n"
                result += "NOT: XOR doğrusal olarak ayrılamaz.\n"
                result += "Tek katmanlı Perceptron XOR'u tam çözemez.\n"
                result += "Bunun için MLP (gizli katmanlı ağ) gerekir.\n"

            self.logic_output_textbox.configure(state="normal")
            self.logic_output_textbox.delete("1.0", "end")
            self.logic_output_textbox.insert("end", result)
            self.logic_output_textbox.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def plot_logic_decision_boundary(self):
        if self.perceptron_model is None:
            messagebox.showwarning("Model Yok", "Önce modeli eğitmelisiniz.")
            return

        Visualizer.plot_logic_gate_decision_boundary(
            perceptron=self.perceptron_model,
            X=self.logic_X,
            y=self.logic_y,
            gate_name=self.selected_gate_name
        )