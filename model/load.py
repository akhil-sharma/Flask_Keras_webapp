from keras.models import load_model
import tensorflow as tf


def init():
    model = load_model('cat_dog_model.h5')
    graph = tf.get_default_graph()
    return model, graph
