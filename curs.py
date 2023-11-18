import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


class SmartphonePricePrediction:
    def __init__(self, model):
        self.model = model #выбираем модель

    def data_loader(self, csv_file):
        df = pd.read_csv(csv_file)   #читаем файл
        y = df['Price']         #колонка цены        
        X = df.drop('Price', axis=1)   # все кроме цены
        X = X.values
        y = y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   #сплитим дату
        return X_train, X_test, y_train, y_test


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)    # учим

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)  # предсказываем несколько 
        return y_pred
    
    def visualizer(self, y_test, y_pred):
        plt.figure(figsize=(10,7))
        plt.scatter(y_test, y_pred)
        plt.plot()
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Scatter Plot')
        plt.show()
        print(f"Mean Squared Error:', {mean_squared_error(y_test, y_pred)}, R2 score: {r2_score(y_test, y_pred)}")


smartphone_model = SmartphonePricePrediction(LinearRegression())

X_train, X_test, y_train, y_test = smartphone_model.data_loader(r"C:\Users\Kravt\OneDrive\Рабочий стол\ML\Cellphone.csv")

smartphone_model.fit(X_train, y_train)

y_pred = smartphone_model.predict_nany(X_test)

print(y_pred, "Test")
print(y_test, "Predicted")
print(smartphone_model.predict_one([[880,10,125,4,233,2,1.3,4,1,3.15,0,1700,9.9]]))

smartphone_model.visualizer(y_test, y_pred)