# 🎓 Student Dropout Prediction App

This is a machine learning web app that predicts whether a student will **Dropout**, **Graduate**, or remain **Enrolled** based on their profile. It uses a trained model and is deployed using **streamli** inside Jupyter Notebook.

---

## 🚀 Features

- Predicts student outcomes using profile features like age, GPA, tuition status, etc.
- Easy-to-use web interface inside Jupyter Notebook (no browser tabs needed).
- Powered by streamlit, Scikit-learn model and pre-processing tools.

---

## 📦 Requirements

Make sure you have Python and Jupyter installed. Then install these packages:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

---

## 📁 Files Needed

Ensure these files are in the same folder:
- `student_dropout_model.pkl` – Trained ML model
- `scaler.pkl` – Feature scaler
- `label_encoder.pkl` – Target encoder

---

## 🧪 Run in Jupyter Notebook

1. Open a Jupyter Notebook.
2. Paste the  app.y code into a cell.
3. At the end of the code, make sure it has:

```python
interface.launch(inline=True)
```

4. Run the cell. The app will appear **inline** inside the notebook.
5. Enter student data and view the prediction result.

---

## 💡 Option: Run as Python Script

If you prefer running it as a normal web app:

```bash
streamlit run app.py
```

Then open the local link (e.g., `http://127.0.0.1:7860`) in your browser.

---

## 🛠 Built With

- Python 🐍
- streamlit
- Scikit-learn 📊
- Jupyter Notebook/ VSC📓

---

## 📄 License

MIT License

---

## 👤 Authors

- Winnie Cheruiyot
- Julian Ayesa
- Delmaine Hoggins
- Harun M. 
- Nice Abere


