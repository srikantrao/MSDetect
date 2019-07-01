"""
Create an LSTM model for classification.
"""

from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.optimizers import Adam
from keras.utils import to_categorical

def create_model(input_shape, drop_rate=0.0):
    
    input_layer = Input(shape=input_shape)
    layer1 = LSTM(256, input_shape=input_shape, return_sequences=True, dropout = drop_rate)(input_layer)
    layer2 = LSTM(256, return_sequences = True, dropout = drop_rate)(layer1)
    layer3 = LSTM(128, dropout = drop_rate)(layer2)
    dense_layer1 = Dense(64)(layer3)
    dense_layer2 = Dense(2, activation='softmax')(dense_layer1)

    return Model(input_layer, dense_layer2)

def  run_model(model, X, y, lr):
    # Compile Model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=lr),
                  metrics=["accuracy"])

    # Fit the modeli
    y_binary = to_categorical(y)
    model.fit(X, y_binary, batch_size = 64, 
              epochs = 100, verbose = 1,
              shuffle = True, validation_split = 0.1)


if __name__ == "__main__":

    model = create_model()

    model.summary()
