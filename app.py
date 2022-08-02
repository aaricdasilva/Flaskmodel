import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('mdl.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('fish_app.html')
@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    Scalar = StandardScaler()
    Scalar.fit(data)
    scaled_int_features = Scalar.transform([int_features])
    final_features = np.array(scaled_int_features)
    prediction = model.predict(final_features)
    if prediction == 0:
        output = 'Bream'
    elif prediction == 1:
        output = 'Roach'
    elif prediction == 2:
        output = 'Whitefish'
    elif prediction == 3:
        output = 'Parkki'
    elif prediction == 4:
        output = 'Perch'
    elif prediction == 5:
        output = 'Pike'
    else:
        output = 'Smelt'

    return render_template('fish_app.html', prediction_text = 'Fish could be {}'.format(output))

if __name__ == "__main__":
    app.run(port=5000, debug=True)
