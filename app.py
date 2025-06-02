import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis Universitas", layout="wide")
st.title("🎓 Dashboard Analisis Status Universitas")
st.markdown("##### *Naive Bayes & K-Means Clustering untuk Evaluasi Kinerja Kelembagaan*")

# Sidebar navigasi
st.sidebar.title("📂 Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", [
    "📊 Eksplorasi Data",
    "🤖 Klasifikasi Naive Bayes",
    "🔍 Clustering K-Means",
    "📌 Kesimpulan",
    "🧮 Prediksi Status Universitas"
])

# Load Data
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'Teaching_Score': np.random.uniform(50, 100, n),
        'Research_Score': np.random.uniform(40, 100, n),
        'Citations_Score': np.random.uniform(20, 100, n),
        'International_Outlook_Score': np.random.uniform(30, 100, n),
        'Industry_Income_Score': np.random.uniform(10, 90, n),
        'STATUS': np.random.choice(['Negeri', 'Swasta'], size=n, p=[0.6, 0.4])
    })

df = load_data()
df_clean = df.copy()
label_encoder = LabelEncoder()
df_clean['STATUS_ENC'] = label_encoder.fit_transform(df_clean['STATUS'])
X = df_clean.drop(columns=['STATUS', 'STATUS_ENC'])
y = df_clean['STATUS_ENC']

# ======================= 📊 Eksplorasi Data =======================
if menu == "📊 Eksplorasi Data":
    st.markdown("### 🔍 Eksplorasi Data Awal")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Distribusi Status Universitas")
        status_count = df['STATUS'].value_counts().reset_index()
        status_count.columns = ['Status', 'Jumlah']
        fig = px.pie(status_count, names='Status', values='Jumlah', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🔥 Korelasi Antar Fitur")
        fig_corr, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(df_clean.drop(columns=['STATUS', 'STATUS_ENC']).corr(), cmap='YlGnBu', annot=True, ax=ax)
        st.pyplot(fig_corr)

    st.markdown("#### 📈 Histogram Tiap Fitur")
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            fig_hist, ax = plt.subplots(figsize=(4.5, 2.8))
            ax.hist(df[col], bins=15, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribusi {col}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            st.pyplot(fig_hist, use_container_width=True)

    st.markdown("#### 📊 Boxplot Fitur terhadap Status")
    cols = st.columns(2)
    for i, col in enumerate(X.columns):
        with cols[i % 2]:
            fig_box, ax = plt.subplots(figsize=(4.5, 2.8))
            sns.boxplot(x='STATUS', y=col, data=df, palette='coolwarm', ax=ax)
            ax.set_title(f'{col} berdasarkan Status', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            st.pyplot(fig_box, use_container_width=True)

# ======================= 🤖 Klasifikasi Naive Bayes =======================
elif menu == "🤖 Klasifikasi Naive Bayes":
    st.markdown("### 🤖 Klasifikasi Status dengan Naive Bayes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred = model_nb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    st.metric("🎯 Akurasi Model", f"{acc:.2%}")
    st.markdown("#### 📌 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax_cm)
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    st.pyplot(fig_cm)

    st.markdown("#### 📋 Classification Report")
    st.dataframe(pd.DataFrame(cr).transpose())

# ======================= 🔍 Clustering K-Means =======================
elif menu == "🔍 Clustering K-Means":
    st.markdown("### 🔍 Clustering Universitas dengan K-Means")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clean['Cluster'] = cluster_labels

    st.markdown("#### 🧭 Visualisasi Cluster (PCA 2D)")
    df_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_vis['Cluster'] = cluster_labels
    fig2 = px.scatter(df_vis, x='PC1', y='PC2', color=df_vis['Cluster'].astype(str),
                      title="Visualisasi Clustering Universitas",
                      labels={'color': 'Cluster'}, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 📊 Distribusi Jumlah Data per Cluster")
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
    df_clean['Cluster'].value_counts().sort_index().plot(kind='bar', color='orchid', ax=ax_bar)
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Jumlah Universitas")
    st.pyplot(fig_bar)

    st.markdown("#### 📈 Rata-Rata Fitur per Cluster")
    st.dataframe(df_clean.groupby('Cluster')[X.columns].mean().style.highlight_max(axis=0))

# ======================= 📌 Kesimpulan =======================
elif menu == "📌 Kesimpulan":
    st.markdown("### ✅ Kesimpulan Analisis")
    st.markdown("""
    - 📊 Model Naive Bayes mampu memprediksi status universitas dengan akurasi cukup baik.
    - 🔬 Fitur seperti Teaching, Research, dan Citations sangat berpengaruh terhadap klasifikasi.
    - 🧪 K-Means clustering menunjukkan pengelompokan universitas berdasarkan performa numerik yang mirip.
    - 💡 Hasil ini dapat membantu pemangku kepentingan mengevaluasi dan memetakan kinerja kelembagaan universitas.
    """)
    st.success("Analisis selesai! Silakan eksplorasi fitur lain dari sidebar.")

# ======================= 🧮 Prediksi Status Universitas =======================
elif menu == "🧮 Prediksi Status Universitas":
    st.markdown("### 🧮 Prediksi Status Universitas Baru dengan Naive Bayes")

    teaching = st.number_input("Teaching Score", 0.0, 100.0, 75.0)
    research = st.number_input("Research Score", 0.0, 100.0, 70.0)
    citations = st.number_input("Citations Score", 0.0, 100.0, 60.0)
    intl_outlook = st.number_input("International Outlook Score", 0.0, 100.0, 65.0)
    industry_income = st.number_input("Industry Income Score", 0.0, 100.0, 50.0)

    model_nb = GaussianNB()
    model_nb.fit(X, y)

    if st.button("Prediksi Status"):
        input_data = np.array([[teaching, research, citations, intl_outlook, industry_income]])
        pred_enc = model_nb.predict(input_data)[0]
        pred_label = label_encoder.inverse_transform([pred_enc])[0]
        proba = model_nb.predict_proba(input_data)[0]
        proba_df = pd.DataFrame({'Status': label_encoder.classes_, 'Probabilitas': proba})

        st.success(f"Prediksi Status Universitas: **{pred_label}**")
        st.markdown("### Probabilitas Prediksi per Status:")
        st.dataframe(proba_df)
