from tensorflow.keras.models import load_model

def predict(customer_id: str):
    model = load_model('models/serialized/lstm_model')
    return model.predict([customer_id])
