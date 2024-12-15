import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Judul Aplikasi
st.title("Model Performance Visualization")

# Input File Data untuk Evaluasi (Opsional)
uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_data.head())

# Dummy Data (Ganti dengan Data Nyata)
y_true = [0, 1, 0, 1, 1, 0, 1, 0]  # Ground truth
y_pred = [0, 1, 0, 1, 0, 0, 1, 1]  # Predicted labels
y_scores = [0.2, 0.8, 0.1, 0.9, 0.4, 0.2, 0.7, 0.6]  # Predicted probabilities (untuk ROC)

# 1. Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues")
st.pyplot(fig)

# 2. Classification Report
st.subheader("Classification Report")
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# 3. ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
st.pyplot(fig)
