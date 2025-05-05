import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np




import subprocess 

def start_analysis():
    ''' Function to open ml.py when the Analysis button is clicked '''
    subprocess.Popen(['python', 'ml.py'])  




FILE_PATH = "students_data.csv"  

def load_csv():
    return pd.read_csv(FILE_PATH)

def preprocess_data(df):
    X = df[['Attendance']] 
    y = df['GPA']  
    pca = PCA(n_components=1) 
    X_pca = pca.fit_transform(X)
    return X_pca, y, X  

def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression - MSE: {mse:.4f}")
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, predictions, color='red', label='Predicted')
    plt.xlabel('Feature: Attendance')
    plt.ylabel('Target: GPA')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

def svm(X_train, y_train, X_test, y_test):
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"SVM (SVR) - MSE: {mse:.4f}")
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, predictions, color='red', label='Predicted')
    plt.xlabel('Feature: Attendance')
    plt.ylabel('Target: GPA')
    plt.title('SVM: Actual vs Predicted')
    plt.legend()
    plt.show()

def decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Decision Tree - MSE: {mse:.4f}")
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, predictions, color='red', label='Predicted')
    plt.xlabel('Feature: Attendance')
    plt.ylabel('Target: GPA')
    plt.title('Decision Tree')
    plt.legend()
    plt.show()

def k_means(X):
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X, y_kmeans, c=y_kmeans, cmap='viridis')
    plt.xlabel('Feature: Attendance')
    plt.ylabel('Cluster')
    plt.title('K-Means Clustering')
    plt.show()

def radial_plot(df):
    department_gpa = df.groupby('Department')['GPA'].mean().reset_index()
    labels = department_gpa['Department'].values
    values = department_gpa['GPA'].values
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Radial Plot (Spider Plot) of Average GPA by Department')
    plt.show()

def heatmap(df):
    correlation_matrix = df[['Attendance', 'GPA']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap: Attendance vs GPA')
    plt.show()

def department_vs_gpa_heatmap(df):
    department_gpa = df.groupby('Department')['GPA'].mean().reset_index()
    plt.figure(figsize=(10, 8))
    sns.heatmap(department_gpa.set_index('Department').T, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title("Department vs GPA Heatmap")
    plt.show()

def pairplot(df):
    sns.pairplot(df)
    plt.title('Pairplot (Scatterplot Matrix)')
    plt.show()


def start_analysism():
    ''' Function to open ml.py when the Analysis button is clicked '''
    subprocess.Popen(['python', 'frontend.py']) 



def run_model(algorithm_choice):
    df = load_csv()
    X, y, X_original = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm_choice == 'Linear Regression':
        linear_regression(X_train, y_train, X_test, y_test)
    elif algorithm_choice == 'SVM':
        svm(X_train, y_train, X_test, y_test)
    elif algorithm_choice == 'Decision Tree':
        decision_tree(X_train, y_train, X_test, y_test)
    elif algorithm_choice == 'K-Means Clustering':
        k_means(X_original)
    elif algorithm_choice == 'Radial Plot':
        radial_plot(df)
    elif algorithm_choice == 'Heatmap':
        heatmap(df)
    elif algorithm_choice == 'Department vs GPA Heatmap':
        department_vs_gpa_heatmap(df)
    elif algorithm_choice == 'Pairplot':
        pairplot(df)
    elif algorithm_choice == 'Return':
        start_analysism()
    else:
        print("Invalid choice")

def create_main_window():
    window = tk.Tk()
    window.title("Data Science Algorithms")

    buttons = [
        ("Linear Regression", "Linear Regression"),
        ("SVM", "SVM"),
        ("Decision Tree", "Decision Tree"),
        ("K-Means Clustering", "K-Means Clustering"),
        ("Radial Plot (Dept vs GPA)", "Radial Plot"),
        ("Heatmap (GPA vs Attendance)", "Heatmap"),
        ("Department vs GPA Heatmap", "Department vs GPA Heatmap"),
        ("Pairplot (Scatterplot Matrix)", "Pairplot"),
        ("Return", "Return")
    ]

    for label, algorithm in buttons:
        button = tk.Button(window, text=label, command=lambda algo=algorithm: run_model(algo))
        button.pack(pady=5)

    window.mainloop()

if __name__ == "__main__":
    create_main_window()
