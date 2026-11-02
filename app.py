"""
Food Grade GUI: photo of a label -> OCR -> editable values -> ML grade (A-E).
Everything runs on the device (no internet needed after install).

Put this file next to train.py, predict.py and the models/ folder.
    pip install rapidocr-onnxruntime pillow opencv-python
    python app.py
"""
import re
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from rapidocr_onnxruntime import RapidOCR

from predict import predict_grade
from train import NUM

COLORS = {"A": "#038141", "B": "#85bb2f", "C": "#fecb02", "D": "#ee8100", "E": "#e63e11"}
LABELS = {"energy-kcal_100g": "Energy (kcal)", "sugars_100g": "Sugar (g)",
          "saturated-fat_100g": "Saturated fat (g)", "salt_100g": "Salt (g)",
          "fiber_100g": "Fibre (g)", "proteins_100g": "Protein (g)"}


# ----------------------------------------------------------------------------
# OCR + parsing
# ----------------------------------------------------------------------------
_ocr = None


def ocr_lines(path):
    """Run OCR and rebuild text rows (so 'Sugar' and '48 g' end up on one line)."""
    global _ocr
    _ocr = _ocr or RapidOCR()
    result, _ = _ocr(path)
    if not result:
        return []
    items = []
    for box, text, _score in result:
        ys, xs = [p[1] for p in box], [p[0] for p in box]
        items.append({"x": min(xs), "yc": sum(ys) / 4, "h": max(ys) - min(ys), "t": text})
    items.sort(key=lambda i: i["yc"])
    rows, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["yc"] - cur[-1]["yc"]) < 0.6 * max(it["h"], cur[-1]["h"]):
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    rows.append(cur)
    return [" ".join(i["t"] for i in sorted(r, key=lambda i: i["x"])) for r in rows]


KEYS = {"sugars_100g": r"sugar", "saturated-fat_100g": r"saturated|sat\.? ?fat",
        "salt_100g": r"\bsalt\b", "fiber_100g": r"fib(?:re|er)", "proteins_100g": r"protein"}
NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")


def _num_after(line, key_re):
    m = re.search(key_re, line, re.I)
    if not m:
        return None
    nums = NUMBER.findall(line[m.end():])
    return float(nums[0].replace(",", ".")) if nums else None


def parse_label(lines):
    """Pull nutrition values + ingredients out of OCR text. Always double-check them."""
    out = {}
    for line in lines:
        low = line.lower()
        if "energy" in low or "calorie" in low:
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*kcal", line, re.I)
            if m:
                out.setdefault("energy-kcal_100g", float(m.group(1).replace(",", ".")))
            else:
                kj = _num_after(line, r"energy|calorie")
                if kj and "kj" in low:
                    out.setdefault("energy-kcal_100g", round(kj / 4.184, 1))
                elif kj:
                    out.setdefault("energy-kcal_100g", kj)
        for key, rx in KEYS.items():
            if key == "sugars_100g" and "added" in low:
                continue
            if key not in out:
                v = _num_after(line, rx)
                if v is not None:
                    out[key] = v
        if "salt_100g" not in out and "sodium" in low:      # sodium -> salt
            v = _num_after(line, r"sodium")
            if v is not None:
                out["salt_100g"] = round(v * 2.5 / (1000 if "mg" in low else 1), 2)

    full = " ".join(lines)
    m = re.search(r"ingredients?\s*[:\-]?\s*(.+)", full, re.I)
    ingredients = ""
    if m:
        ingredients = re.split(r"nutrition|nutritional|contains|allergen|may contain|"
                               r"manufactured|best before|storage", m.group(1), flags=re.I)[0].strip()
    return out, ingredients, "\n".join(lines)


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Food Grade Scanner")
        self.geometry("980x640")
        self.status = tk.StringVar(value="Open a photo of the nutrition label to start.")
        self.entries = {}
        self._build()

    def _build(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        self.preview = ttk.Label(left, text="No image", width=48, anchor="center")
        self.preview.pack(pady=(0, 8))
        ttk.Button(left, text="Open photo", command=self.open_photo).pack(fill="x", pady=2)
        ttk.Button(left, text="Capture from webcam", command=self.webcam).pack(fill="x", pady=2)
        ttk.Label(left, text="OCR text (to check)").pack(anchor="w", pady=(10, 0))
        self.raw = tk.Text(left, width=44, height=14, font=("Consolas", 9))
        self.raw.pack()

        right = ttk.Frame(self, padding=10)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Values per 100 g (edit if OCR got them wrong)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        grid = ttk.Frame(right)
        grid.pack(anchor="w", pady=6)
        for r, key in enumerate(NUM):
            ttk.Label(grid, text=LABELS[key], width=18).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(grid, width=10)
            e.grid(row=r, column=1)
            self.entries[key] = e

        ttk.Label(right, text="Ingredients").pack(anchor="w", pady=(8, 0))
        self.ingredients = tk.Text(right, height=5, wrap="word")
        self.ingredients.pack(fill="x")
        ttk.Button(right, text="Get grade", command=self.predict).pack(anchor="w", pady=10)

        strip = ttk.Frame(right)
        strip.pack(anchor="w")
        self.badges = {}
        for g in "ABCDE":
            l = tk.Label(strip, text=g, width=3, height=1, fg="white", bg="#cfd8dc",
                         font=("Segoe UI", 20, "bold"))
            l.pack(side="left", padx=3)
            self.badges[g] = l
        self.result = ttk.Label(right, text="", wraplength=480, justify="left")
        self.result.pack(anchor="w", pady=8)
        ttk.Label(right, textvariable=self.status, foreground="#5d6e77").pack(side="bottom", anchor="w")

    # ---- image input ----------------------------------------------------------
    def open_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self.load_image(path)

    def webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open the webcam.")
            return
        frame = None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imshow("SPACE = capture, ESC = cancel", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                frame = img
                break
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame is not None:
            path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
            cv2.imwrite(path, frame)
            self.load_image(path)

    def load_image(self, path):
        img = Image.open(path)
        img.thumbnail((360, 360))
        self._photo = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._photo, text="")
        self.status.set("Reading label (first run is slower)...")
        threading.Thread(target=self._ocr_worker, args=(path,), daemon=True).start()

    def _ocr_worker(self, path):
        try:
            data = parse_label(ocr_lines(path))
            self.after(0, self.fill, *data)
        except Exception as e:                                  # show errors in the UI
            self.after(0, lambda: self.status.set(f"OCR failed: {e}"))

    def fill(self, values, ingredients, raw):
        for key, e in self.entries.items():
            e.delete(0, "end")
            if key in values:
                e.insert(0, str(values[key]))
        self.ingredients.delete("1.0", "end")
        self.ingredients.insert("1.0", ingredients)
        self.raw.delete("1.0", "end")
        self.raw.insert("1.0", raw)
        self.status.set(f"Found {len(values)}/6 nutrition values. Check them, then press Get grade.")

    # ---- prediction -----------------------------------------------------------
    def predict(self):
        nutrition = {}
        for key, e in self.entries.items():
            t = e.get().strip().replace(",", ".")
            if t:
                try:
                    nutrition[key] = float(t)
                except ValueError:
                    messagebox.showerror("Invalid value", f"{LABELS[key]} must be a number.")
                    return
        try:
            res = predict_grade(nutrition=nutrition, ingredients=self.ingredients.get("1.0", "end"))
        except ValueError as e:
            messagebox.showinfo("Not enough data", str(e))
            return
        for g, lab in self.badges.items():
            on = g == res["grade"]
            lab.configure(bg=COLORS[g] if on else "#cfd8dc",
                          font=("Segoe UI", 34 if on else 20, "bold"))
        why = "\n".join("• " + r for r in res["reasons"])
        self.result.configure(text=f"Grade {res['grade']}  (confidence {res['confidence']:.0%})\n"
                                   f"Model: {res['method']}\n{why}")


if __name__ == "__main__":
    App().mainloop()
