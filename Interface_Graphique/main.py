import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import pandas as pd
import subprocess
import threading
import time

# ====================== Constantes ======================
IMG_WIDTH, IMG_HEIGHT = 200, 250
CONFIDENCE_THRESHOLD = 0.7

# ====================== Fonctions utilitaires ======================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Impossible de lire l'image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_preprocessed = tf.keras.applications.resnet.preprocess_input(img_resized.astype(np.float32))
    return np.expand_dims(img_preprocessed, axis=0)

def classify_image(model, class_names, img_path):
    img_tensor = preprocess_image(img_path)
    preds = model.predict(img_tensor)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "inconnu", confidence
    return class_names[class_idx], confidence

# ====================== Classe App ======================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Classificateur & Entraînement d'Images")
        root.geometry("1000x750")

        self.model = None
        self.class_names = []
        self.model_path = ""
        self.results = []

        self.tab_control = ttk.Notebook(root)
        self.tab_classify = ttk.Frame(self.tab_control)
        self.tab_train = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_classify, text="Classification")
        self.tab_control.add(self.tab_train, text="Réentraînement")
        self.tab_control.pack(expand=1, fill="both")

        self.setup_classification_tab()
        self.setup_retrain_tab()

    # ---------------------- Onglet Classification ----------------------
    def setup_classification_tab(self):
        frame_model = tk.Frame(self.tab_classify)
        frame_model.pack(pady=10)
        tk.Button(frame_model, text="Charger Modèle", command=self.load_model).pack(side=tk.LEFT)
        self.label_model_name = tk.Label(frame_model, text="Aucun modèle chargé", fg="red")
        self.label_model_name.pack(side=tk.LEFT, padx=10)

        frame_btns = tk.Frame(self.tab_classify)
        frame_btns.pack()
        tk.Button(frame_btns, text="Image Unique", command=self.upload_image).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_btns, text="Images Multiples", command=self.upload_multiple_images).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_btns, text="Dossier", command=self.upload_folder).pack(side=tk.LEFT, padx=5)

        self.image_label = tk.Label(self.tab_classify)
        self.image_label.pack(pady=10)
        self.result_label = tk.Label(self.tab_classify, text="", font=("Arial", 12))
        self.result_label.pack()

        columns = ('Fichier', 'Classe', 'Confiance')
        self.tree = ttk.Treeview(self.tab_classify, columns=columns, show='headings', height=15)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center', width=200)
        self.tree.pack(pady=10, fill="x")

        tk.Button(self.tab_classify, text="Exporter CSV", command=self.export_csv).pack()
        tk.Button(self.tab_classify, text="Réinitialiser", command=self.reset_all).pack(pady=5)
        self.stats_label = tk.Label(self.tab_classify, text="", font=("Arial", 11))
        self.stats_label.pack(anchor="w", padx=10)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Modèles Keras", "*.h5")])
        if not path:
            return
        self.model = tf.keras.models.load_model(path)
        self.model_path = path
        n_classes = self.model.output_shape[-1]
        self.class_names = ['crocodile', 'hyène', 'léopard', 'lion'] if n_classes == 4 else ['crocodile', 'hyène', 'léopard', 'lion', 'lycaons']
        self.label_model_name.config(text=os.path.basename(path), fg="green")
        self.results.clear()
        self.update_table()

    def upload_image(self):
        if not self.model:
            messagebox.showwarning("Alerte", "Chargez un modèle d'abord")
            return
        path = filedialog.askopenfilename()
        if path:
            self.display_image(path)
            pred, conf = classify_image(self.model, self.class_names, path)
            self.results.append((os.path.basename(path), pred, f"{conf*100:.2f}%"))
            self.result_label.config(text=f"Prédiction : {pred} ({conf*100:.2f}%)")
            self.update_table()

    def upload_multiple_images(self):
        paths = filedialog.askopenfilenames()
        if paths:
            for path in paths:
                try:
                    pred, conf = classify_image(self.model, self.class_names, path)
                    self.results.append((os.path.basename(path), pred, f"{conf*100:.2f}%"))
                except:
                    self.results.append((os.path.basename(path), "Erreur", "0"))
            self.update_table()

    def upload_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isfile(path):
                    try:
                        pred, conf = classify_image(self.model, self.class_names, path)
                        self.results.append((file, pred, f"{conf*100:.2f}%"))
                    except:
                        self.results.append((file, "Erreur", "0"))
            self.update_table()

    def export_csv(self):
        if not self.results:
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv")
        if file:
            pd.DataFrame(self.results, columns=["Fichier", "Classe", "Confiance"]).to_csv(file, index=False)
            messagebox.showinfo("Succès", f"Exporté vers {file}")

    def reset_all(self):
        self.results.clear()
        self.update_table()
        self.result_label.config(text="")
        self.image_label.config(image='')

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        self.img_display = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.img_display)

    def update_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for r in self.results:
            self.tree.insert('', tk.END, values=r)
        self.update_stats()

    def update_stats(self):
        counts = {}
        for _, c, _ in self.results:
            counts[c] = counts.get(c, 0) + 1
        stat_txt = "Statistiques :\n" + "\n".join([f"{k} : {v}" for k, v in counts.items()])
        self.stats_label.config(text=stat_txt)

    # ---------------------- Onglet Réentraînement ----------------------
    def setup_retrain_tab(self):
        frame = self.tab_train

        tk.Label(frame, text="Dossier d'entraînement").pack()
        self.train_dir = tk.Entry(frame, width=60)
        self.train_dir.pack()
        tk.Button(frame, text="Parcourir", command=lambda: self.select_dir(self.train_dir)).pack()

        tk.Label(frame, text="Dossier de validation").pack()
        self.val_dir = tk.Entry(frame, width=60)
        self.val_dir.pack()
        tk.Button(frame, text="Parcourir", command=lambda: self.select_dir(self.val_dir)).pack()

        tk.Label(frame, text="Époques").pack()
        self.epochs_var = tk.IntVar(value=100)
        tk.Entry(frame, textvariable=self.epochs_var).pack()

        self.device = tk.StringVar(value="gpu")
        tk.Radiobutton(frame, text="GPU", variable=self.device, value="gpu").pack()
        tk.Radiobutton(frame, text="CPU", variable=self.device, value="cpu").pack()

        tk.Label(frame, text="Fichier de sortie du modèle").pack()
        self.save_model_path = tk.Entry(frame, width=60)
        self.save_model_path.pack()
        tk.Button(frame, text="Choisir fichier", command=self.select_save_path).pack()

        self.train_btn = tk.Button(frame, text="Lancer l'entraînement", command=self.train_model)
        self.train_btn.pack(pady=10)

        self.progress = ttk.Progressbar(frame, orient='horizontal', length=400, mode='determinate')
        self.progress.pack()

        self.output_text = tk.Text(frame, height=15, width=100)
        self.output_text.pack(pady=10)

    def select_dir(self, entry):
        d = filedialog.askdirectory()
        if d:
            entry.delete(0, tk.END)
            entry.insert(0, d)

    def select_save_path(self):
        f = filedialog.asksaveasfilename(defaultextension=".h5")
        if f:
            self.save_model_path.delete(0, tk.END)
            self.save_model_path.insert(0, f)

    def train_model(self):
        train_dir = self.train_dir.get()
        val_dir = self.val_dir.get()
        epochs = self.epochs_var.get()
        device = self.device.get()
        save_path = self.save_model_path.get()

        if not all([train_dir, val_dir, save_path]):
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
            return

        self.output_text.delete(1.0, tk.END)
        self.progress['value'] = 0

        cmd = [
            sys.executable, 'training.py',
            '--train_dir', train_dir,
            '--validation_dir', val_dir,
            '--save_path', save_path,
            '--epochs', str(epochs),
            '--device', device
        ]

        def run():
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in iter(process.stdout.readline, ''):
                    self.output_text.insert(tk.END, line)
                    self.output_text.see(tk.END)
                    if 'Epoch' in line and '/' in line:
                        parts = line.strip().split('/')
                        if len(parts) >= 2:
                            current = int(parts[0].split()[-1])
                            total = int(parts[1].split()[0])
                            self.progress['value'] = (current / total) * 100
                process.stdout.close()
                process.wait()
            except Exception as e:
                messagebox.showerror("Erreur", str(e))

        threading.Thread(target=run).start()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
