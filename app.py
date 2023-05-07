from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load("best_model.pkl")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input variables
    var1 = int(request.json['var1'])
    var2 = int(request.json['var2'])
    var3 = int(request.json['var3'])
    var4 = int(request.json['var4'])
    var5 = int(request.json['var5'])
    var6 = int(request.json['var6'])
    var7 = int(request.json['var7'])
    var8 = int(request.json['var8'])
    var9 = int(request.json['var9'])

    # Make a prediction using the machine learning model
    prediction = model.predict([[var1, var2, var3, var4, var5, var6, var7, var8, var9]])

    print(var1, var2, var4, var5, var7, var8, var9)

    # Convert the prediction to a string
    if prediction[0] == 0:
        pred_string = "Not depressed"
        print(pred_string)
    else:
        pred_string = "Depressed"
        print(pred_string)

    # Return the prediction as a JSON object
    return jsonify({'prediction': pred_string})


if __name__ == '__main__':
    app.run(debug=True)
