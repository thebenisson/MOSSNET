import tensorflow as tf
import pathlib
from utils import Params
from model.hardTripletLoss import TripletHardLoss
from model.mossnet import CreateModel
from model.dataset import train_data_loader

params = Params() # load hyperparameters
checkpoint_path = pathlib.Path("data/training/cp.ckpt")
save_model_path = pathlib.Path("data/saved_model/")

if __name__ == "__main__":
    try:
        # load model so it can be trained further
        mossnet = tf.keras.models.load_model('data/saved_model/MOSSNET',
                    custom_objects={'hard_triplet_loss': TripletHardLoss()})
    except Exception as exp:
        pass

    # create new model if trained model doesn't exist
    if !mossnet:
        mossnet = CreateModel()

    # configure model with optimizer and cost funtion for training
    mossnet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=params.learning_rate),
                loss=TripletHardLoss(margin=params.margin))

    # create callback so traning auto checkpoint after each epoch 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path)
    # set up training dataset generator
    train_gen = train_data_loader()
    print(f'Starting training for {params.epochs} epochs')

    # train model with tf.keras' model.fit API
    learning = mossnet.fit(train_gen, epochs=params.epochs, callbacks=[checkpoint_callback])
    # save model once training is complete so it can be used for prediction at any time
    mossnet.save(save_model_path)
    print(f'Training complete, model saved at {save_model_path}')
