import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


class WeatherPredictor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self.extract()
        self.column_name = ["Rh1", "Rh2", "Tmax", "Tmin", "Rainfall"]
        self.n_past = 30
        self.n_fut = 1

    def extract(self):
        return pd.read_csv(self.filepath)

    def fill_missing_value_with_avg(self):
        for col in self.column_name:
            for i in range(1, len(self.df[col]) - 1):
                if pd.isnull(self.df.loc[i, col]):
                    above_value = self.df.loc[i - 1, col]
                    below_value = self.df.loc[i + 1, col]
                    if pd.notnull(above_value) and pd.notnull(below_value):
                        self.df.loc[i, col] = (above_value + below_value) / 2

    def clean_rainfall_column(self):
        self.df['Rainfall'] = self.df['Rainfall'].replace('T', 0.01).astype(float)

    def categorize_rainfall(self, value):
        return 'No Rain' if value <= 0 else 'Rain'

    def apply_rainfall_categorization(self):
        self.df['Rain_Category'] = self.df['Rainfall'].apply(self.categorize_rainfall)

    def encode_rain_category(self):
        category_mapping = {'No Rain': 0, 'Rain': 1}
        self.df['Rain_Category_Encoded'] = self.df['Rain_Category'].map(category_mapping).astype(int)

    def split_train_test(self):
        df = self.df.drop(["Rain_Category", "Rainfall", "Date"], axis=1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = [], []

        for i in range(self.n_past, len(scaled_data) - self.n_fut + 1):
            X.append(scaled_data[i - self.n_past:i, 0:df.shape[1]])
            y.append(scaled_data[i + self.n_fut - 1:i + self.n_fut, -1])

        X, y = np.array(X), np.array(y)
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def build_lstm_model(self, input_shape):
        model = keras.Sequential()
        model.add(layers.LSTM(20, activation='tanh', use_bias=True, input_shape=input_shape, return_sequences=True))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.LSTM(10, activation='tanh', use_bias=True, input_shape=input_shape))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit_model_with_early_stopping(self, model, trainX, trainY):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        return model.fit(trainX, trainY, shuffle=False, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    def save_loss_plot(self, history, filename):
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def run(self):
        self.fill_missing_value_with_avg()
        self.clean_rainfall_column()
        self.apply_rainfall_categorization()
        self.encode_rain_category()

        X_train, X_test, y_train, y_test = self.split_train_test()
        np.save('X_test.npy', X_test)
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        history = self.fit_model_with_early_stopping(model, X_train, y_train)

        self.save_loss_plot(history, "output_loss.png")

        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        conf_mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        model.save('my_model.h5')

if __name__ == "__main__":
    predictor = WeatherPredictor("RainfallDataset.csv")
    predictor.run()
