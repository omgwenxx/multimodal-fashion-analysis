## 🚀 How to Run the Streamlit App

This guide explains how to set up and launch a [Streamlit](https://streamlit.io) application locally.

### 📦 1. Install Streamlit

```bash
pip install streamlit=1.34.0
```

---

### 📁 2. Navigate to the App Directory

```bash
cd image_captioning/app
```

---

### ▶️ 3. Run the App

Use the following command to start the Streamlit server for the FIC dataset:

```bash
streamlit run display_samples_fic.py
```

Or for the H&M dataset.:

```bash
streamlit run display_samples_hm.py
```

---

### 🌐 4. Open in Browser

After starting, Streamlit will launch the app in your default browser at:

```
http://localhost:8501
```

If it doesn’t open automatically, you can manually visit the URL above. If you are using remote development like I was be sure to port forward.

---

### 🧼 5. Stop the App

To stop the app, press `Ctrl+C` in the terminal.
