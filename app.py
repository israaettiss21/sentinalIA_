from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)


def preprocessing(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    df = df.sample(frac=1)  # Shuffle the DataFrame
    x = df.iloc[:, df.columns != 'Label']
    y = df[['Label']].to_numpy()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    return x, y


def reshape_dataset_cnn(x: np.ndarray) -> np.ndarray:
    result = np.zeros((x.shape[0], 81))
    result[:, :-3] = x
    result = np.reshape(result, (result.shape[0], 9, 9))
    result = result[..., np.newaxis]
    return result


model = load_model('SentinalIA_cnn.h5')
test_data = pd.read_csv('testappsentinalIA.csv')

labelencoder = LabelEncoder()
labelencoder.classes_ = np.load('label_encoder (1).npy', allow_pickle=True)

X_test, y_test = preprocessing(test_data)
X_test_cnn = reshape_dataset_cnn(X_test)

predictions = []

@app.route('/')
def index():
    return render_template('index.html', attack=predictions[0] if len(predictions)>0 else "No Attacks", action="Take Action")

@app.route('/start')
def start_stream():
    for i in range(X_test_cnn.shape[0]):
        sample = X_test_cnn[i:i+1]  
        prediction = model.predict(sample)  
        predicted_class = np.argmax(prediction) 
        original_label = labelencoder.inverse_transform([predicted_class])[0]
        predictions.append(original_label)
        print(f"Sample {i+1} - Predicted Class: {original_label}")
        time.sleep(3)  # Sleep for 3 seconds to simulate streaming interval
    return "Prediction ended"

@app.route('/next')
def next_prediction():
    return predictions.pop(0) if len(predictions)>0 else "No attacks"

if __name__ == '__main__':
    app.run(debug=True, port=8080)