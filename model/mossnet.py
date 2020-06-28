from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from utils import Params

params = Params() # load hyperparameters

def CreateModel(emb_size=params.embedding_size):
    base_model = ResNet50(input_shape=(params.img_height,params.img_width, params.channels),
                                   pooling='max', include_top=False)
    base_model.trainable = True
    mossnet = Sequential()
    mossnet.add(base_model, name='base_layer')
    mossnet.add(Dense(emb_size), name='embedding_layer')
    mossnet.add(Lambda(lambda x : tf.math.l2_normalize(x,axis=1), name='embeddings_l2norm'))
    return mossnet