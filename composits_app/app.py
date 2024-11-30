from flask import Flask, render_template, request, jsonify
import numpy as np
from static.src.prediction import tensile_properties_prediction


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_features = np.array([
        float(data['matrix_filler_ratio']),
        float(data['density']),
        float(data['elasticity_modulus']),
        float(data['hardener_amount']),
        float(data['epoxy_groups_amount']),
        float(data['flash_point']),
        float(data['surface_density']),
        float(data['resin_consumption']),
        float(data['patch_angle']),
        float(data['patch_pitch']),
        float(data['patch_density'])
    ]).reshape(1,-1)
    
    model = tensile_properties_prediction()
    predicted_values = model.predict(input_features)

    result = jsonify({
        'tensile_modulus_of_elasticity' : str(predicted_values[0][0]),
        'tensile_strength': str(predicted_values[0][1])
    })

    return result, 200, {'Content-Type': 'application/json  '}


@app.route('/about/')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)