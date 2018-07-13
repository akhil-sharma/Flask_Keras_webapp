from keras.models import load_model

def init():
    model = load_model('model.h5')
    return model