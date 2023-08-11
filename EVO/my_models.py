import keras
from tensorflow.keras.layers import Dense, GRU, Dropout, Flatten
from keras import regularizers
from keras.models import Sequential

from loss_functions import top_is_target_31



def gru_simple_v1_model(X_DATA, Y_DATA):
    inputs = keras.Input(shape=(X_DATA.shape[1:]))
    m = inputs

    mA = GRU(units=1, kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(m)

    mA = Dropout(0.1)(mA)

    mA = Dense(units=1, activation = 'relu')(mA)

    m = Flatten()(mA)

    # Añadir capa Dense de salida
    out = Dense(Y_DATA.shape[1], activation='tanh')(m)

    model_GRU = keras.Model(inputs=inputs, outputs=out)

    # Compilar el modelo
    model_GRU.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=top_is_target_31,
            metrics=[])

    return model_GRU
