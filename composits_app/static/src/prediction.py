import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class tensile_properties_prediction():
    def __init__(self):
        self.model = tf.keras.models.load_model('composits_app/static/models/nn_stregth_elasticity.keras')
        with open('composits_app/static/models/scalers.save', 'rb') as f:
            self.features_scaler, self.targets_scaler = joblib.load(f)

    def predict(self, features):
        _features = self.features_scaler.transform(features)
        prediction = self.model.predict(_features)
        prediction = self.targets_scaler.inverse_transform(prediction)
        
        return prediction
