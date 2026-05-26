import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(df, X_scaled):
    y = df['status_lulus']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model_lr = LogisticRegression()
    model_knn = KNeighborsClassifier(n_neighbors=5)
    
    model_lr.fit(X_train, y_train)
    model_knn.fit(X_train, y_train)
    
    return model_lr, model_knn, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Terlambat', 'Tepat Waktu'], 
                yticklabels=['Terlambat', 'Tepat Waktu'])
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.title(title)
    return fig
