from sklearn.model_selection import train_test_split
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from utils import make_data_pipeline,azim_equidist_projection
import keras
from tensorflow_addons.activations import sparsemax
from tensorflow.keras.layers import Dropout, Flatten, Embedding, LSTM, Bidirectional, multiply, Conv2D, GlobalMaxPooling1D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model
from model import build_att_cnn_model,build_cnn_model,make_bilstm_model,make_atten_bilstm_model,make_atten1_bilstm_model

locs = scipy.io.loadmat('/input/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_equidist_projectio(e))
       
file_names = []
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        file = os.path.join(dirname, filename)
        file_names.append(file)

labels = np.asarray(pd.read_csv("/input/......."))
image_size = 32
frame_duration = 1
overlap = 0.8
X, y, X_0t = make_data_pipeline(file_names,labels,image_size,frame_duration,overlap)

Print(X.shape,X_0t.shape)
Print(y.shape)

x_train, x_test, y_train, y_test, x_eeg_train, x_eeg_test = train_test_split(X, y,X_0t, test_size=0.20,shuffle=True)

# input image dimensions
img_rows, img_cols = image_size, image_size

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape = (img_rows, img_cols, 3)

batch_size = 64
num_classes = 2

maxlen = X_0t.shape[1]

max_features = X_0t.max()+1

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#active below block if want to run on a TPU server
'''
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
'''

with strategy.scope():
    atten_cnn = build_att_cnn_model()
    atten_bilstm = make_atten_bilstm_model()
    x = multiply([cnn.output, atten_bilstm.output])
    #x = build_cnn_model()
    #x = make_bilstm_model()
    predictions = Dense(num_classes)(x)
    predictions = sparsemax(predictions)
    #predictions = Dense(num_classes, activation='softmax')(cnn.output)
    #predictions = Dense(num_classes, activation='softmax')(atten_bilstm.output)
    model = Model(inputs=[cnn.input, atten_bilstm.input], outputs=predictions)
    #model = Model(inputs=atten_bilstm.input, outputs=predictions)
    #model = Model(inputs=cnn.input, outputs=predictions)
    #flatten = model.Flatten()
    #dense3 = model.Dense(512)
    #act3 = model.Activation("relu")
    #bn3 = model.BatchNormalization()
    #do3 = model.Dropout(0.5)
    # initialize the layers in the softmax classifier layer set
    #dense4 = model.Dense(classes)
    #softmax = model.Activation("softmax")

opt = keras.optimizers.Adam(lr=0.0001,decay=1e-3 / 200, epsilon=1e-07)

model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
print(model.summary())

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_eeg_train = x_eeg_train.astype('float32')
x_eeg_test = x_eeg_test.astype('float32')
#x_train /= 255
#x_test /= 255
x_train.shape,y_train.shape,x_eeg_train.shape,x_eeg_test.shape,y_test.shape,x_test.shape



es = EarlyStopping(monitor='val_los', mode='min', verbose=1, patience=500)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(x=[x_train,x_eeg_train], y = y_train,
          batch_size=batch_size,
          epochs=2000,
          validation_data=([x_test,x_eeg_test], y_test),
          shuffle=True,verbose=1, callbacks=[es, mc])
          
          
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



saved_model = model.load_weights('best_model.h5')

_, train_acc = model.evaluate([x_train,x_eeg_train], y_train, verbose=0)
_, test_acc = model.evaluate([x_test,x_eeg_test], y_test, verbose=0)

print('Train: %.5f, Test: %.5f' % (train_acc, test_acc))
                  
