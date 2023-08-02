

from keras import backend as K
import tensorflow as tf

def top_is_target(y_true, y_pred, hp_top = 10, hp_incremento_top = 2):
  num_samples = K.shape(y_true)[0]

  w = K.arange(31, dtype='float32')
  w = K.reverse(w, axes=0) + 1

  w = tf.where(K.arange(31) < hp_top, w * hp_incremento_top, w)
  # Replicar w a lo largo del eje 0 (batch)
  #w = K.repeat_elements(K.expand_dims(w, 0), num_samples, axis=0)
  w = tf.tile(K.expand_dims(w, 0), [num_samples, 1])

  # Calcular la pérdida
  r = K.cast(y_true, 'float32')
  r_pred = K.cast(y_pred, 'float32')

  return K.sum(w * K.square(r - r_pred), axis=-1)