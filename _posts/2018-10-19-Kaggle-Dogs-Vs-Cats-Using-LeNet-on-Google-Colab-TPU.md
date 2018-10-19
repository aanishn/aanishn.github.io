
Kaggle Dogs Vs. Cats Using LeNet  on Google Colab TPU
==================================================

### Required setup

1. Update api_token with kaggle api key for downloading dataset

          - Login to kaggle
          - My Profile > Edit Profile > Createt new API Token
          - Update **api_token** dict below with the values
          
2. Change Notebook runtime to TPU
          
          - In colab notebook menu, Runtime > Change runtime type
          - Select TPU in the list

Install kaggle package, download and extract zip file


```
!pip install kaggle

api_token = {"username":"xxxxx","key":"xxxxxxxxxxxxxxxxxxxxxxxx"}

import json
import zipfile
import os

os.mkdir('/root/.kaggle')

with open('/root/.kaggle/kaggle.json', 'w') as file:
    json.dump(api_token, file)
!chmod 600 /root/.kaggle/kaggle.json
# !kaggle config path -p /root
!kaggle competitions download -c dogs-vs-cats
```

    Collecting keras
    [?25l  Downloading https://files.pythonhosted.org/packages/5e/10/aa32dad071ce52b5502266b5c659451cfd6ffcbf14e6c8c4f16c0ff5aaab/Keras-2.2.4-py2.py3-none-any.whl (312kB)
    [K    100% |████████████████████████████████| 317kB 9.7MB/s 
    [?25hRequirement already satisfied, skipping upgrade: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.6/dist-packages (from keras) (1.0.5)
    Requirement already satisfied, skipping upgrade: six>=1.9.0 in /usr/local/lib/python3.6/dist-packages (from keras) (1.11.0)
    Requirement already satisfied, skipping upgrade: h5py in /usr/local/lib/python3.6/dist-packages (from keras) (2.8.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.9.1 in /usr/local/lib/python3.6/dist-packages (from keras) (1.14.6)
    Requirement already satisfied, skipping upgrade: scipy>=0.14 in /usr/local/lib/python3.6/dist-packages (from keras) (0.19.1)
    Requirement already satisfied, skipping upgrade: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from keras) (1.0.6)
    Requirement already satisfied, skipping upgrade: pyyaml in /usr/local/lib/python3.6/dist-packages (from keras) (3.13)
    Installing collected packages: keras
      Found existing installation: Keras 2.1.6
        Uninstalling Keras-2.1.6:
          Successfully uninstalled Keras-2.1.6
    Successfully installed keras-2.2.4
    Collecting kaggle
    [?25l  Downloading https://files.pythonhosted.org/packages/c6/78/832b9a9ec6b3baf8ec566e1f0a695f2fd08d2c94a6797257a106304bfc3c/kaggle-1.4.7.1.tar.gz (52kB)
    [K    100% |████████████████████████████████| 61kB 6.5MB/s 
    [?25hRequirement already satisfied: urllib3<1.23.0,>=1.15 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.22)
    Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.11.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2018.10.15)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.5.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.18.4)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.27.0)
    Collecting python-slugify (from kaggle)
      Downloading https://files.pythonhosted.org/packages/00/ad/c778a6df614b6217c30fe80045b365bfa08b5dd3cb02e8b37a6d25126781/python-slugify-1.2.6.tar.gz
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.6)
    Collecting Unidecode>=0.04.16 (from python-slugify->kaggle)
    [?25l  Downloading https://files.pythonhosted.org/packages/59/ef/67085e30e8bbcdd76e2f0a4ad8151c13a2c5bce77c85f8cad6e1f16fb141/Unidecode-1.0.22-py2.py3-none-any.whl (235kB)
    [K    100% |████████████████████████████████| 235kB 11.3MB/s 
    [?25hBuilding wheels for collected packages: kaggle, python-slugify
      Running setup.py bdist_wheel for kaggle ... [?25l- \ done
    [?25h  Stored in directory: /root/.cache/pip/wheels/44/2c/df/22a6eeb780c36c28190faef6252b739fdc47145fd87a6642d4
      Running setup.py bdist_wheel for python-slugify ... [?25l- done
    [?25h  Stored in directory: /root/.cache/pip/wheels/e3/65/da/2045deea3098ed7471eca0e2460cfbd3fdfe8c1d6fa6fcac92
    Successfully built kaggle python-slugify
    Installing collected packages: Unidecode, python-slugify, kaggle
    Successfully installed Unidecode-1.0.22 kaggle-1.4.7.1 python-slugify-1.2.6
    Downloading sampleSubmission.csv to /content
      0% 0.00/86.8k [00:00<?, ?B/s]
    100% 86.8k/86.8k [00:00<00:00, 37.5MB/s]
    Downloading test1.zip to /content
     95% 257M/271M [00:01<00:00, 111MB/s]
    100% 271M/271M [00:02<00:00, 137MB/s]
    Downloading train.zip to /content
     97% 529M/543M [00:03<00:00, 198MB/s]
    100% 543M/543M [00:03<00:00, 159MB/s]



```
zip_ref = zipfile.ZipFile('/content/train.zip', 'r')
zip_ref.extractall()
zip_ref.close()
```

Re-arrange classes to 2 separate directories


```
!mkdir train/cat train/dog
!mv train/*cat*.jpg train/cat
!mv train/*dog*.jpg train/dog
```

Training configs


```
BATCH_SIZE   = 64
IMG_DIM      = (256, 256, 3)
NUM_EPOCHS   = 1
```

Setup generators to provide with train and validation batches


```
import tensorflow as tf
from tensorflow import keras

print(keras.__version__)
print(tf.__version__)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

traingen = datagen.flow_from_directory(
    'train',
    batch_size = BATCH_SIZE,
    target_size = IMG_DIM[:-1],
    class_mode = 'categorical',
    subset='training')

valgen = datagen.flow_from_directory(
    'train',
    batch_size = BATCH_SIZE,
    target_size = IMG_DIM[:-1],
    class_mode = 'categorical',
    subset='validation')
```

    2.1.6-tf
    1.11.0
    Found 20000 images belonging to 2 classes.
    Found 5000 images belonging to 2 classes.


Define LeNet model architecture


```
input = keras.layers.Input(IMG_DIM, name="input")
conv1 = keras.layers.Conv2D(20, kernel_size=(5, 5), padding='same')(input)
pool1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
conv2 = keras.layers.Conv2D(50, kernel_size=(5,5), padding='same')(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
flatten1 = keras.layers.Flatten()(pool2)
fc1 = keras.layers.Dense(500, activation='relu')(flatten1)
fc2 = keras.layers.Dense(2, activation='softmax')(fc1)

model = keras.models.Model(inputs=input, outputs=fc2)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=0.01),
    metrics=['accuracy'])

print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input (InputLayer)           (None, 256, 256, 3)       0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 256, 256, 20)      1520      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 128, 128, 20)      0         
    _________________________________________________________________
    flatten (Flatten)            (None, 327680)            0         
    _________________________________________________________________
    dense (Dense)                (None, 500)               163840500 
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 1002      
    =================================================================
    Total params: 163,843,022
    Trainable params: 163,843,022
    Non-trainable params: 0
    _________________________________________________________________
    None


Check for TPU availability


```
import os

try:
  device_name = os.environ['COLAB_TPU_ADDR']
  TPU_ADDRESS = 'grpc://' + device_name
  print('Found TPU at: {}'.format(TPU_ADDRESS))

except KeyError:
  print('TPU not found')
```

    Found TPU at: grpc://10.73.36.106:8470


Convert keras model to TPU model


```
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)))
```

    INFO:tensorflow:Querying Tensorflow master (b'grpc://10.73.36.106:8470') for TPU system metadata.
    INFO:tensorflow:Found TPU system:
    INFO:tensorflow:*** Num TPU Cores: 8
    INFO:tensorflow:*** Num TPU Workers: 1
    INFO:tensorflow:*** Num TPU Cores Per Worker: 8
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, -1, 4430250600885613814)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 17179869184, 14860921772671154020)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_GPU:0, XLA_GPU, 17179869184, 10329331434607546216)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 17179869184, 3020471452782936925)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 17179869184, 7058080726911325303)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 17179869184, 616613029749391519)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 17179869184, 11136912683004452860)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 17179869184, 4887088926811133454)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 17179869184, 2967281305396391452)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 17179869184, 287043896194918914)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 17179869184, 7338422145717968905)
    INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 17179869184, 11776534234889956760)
    WARNING:tensorflow:tpu_model (from tensorflow.contrib.tpu.python.tpu.keras_support) is experimental and may change or be removed at any time, and without warning.
    INFO:tensorflow:Connecting to: b'grpc://10.73.36.106:8470'


Run training


```
tpu_model.fit_generator(
    traingen,
    steps_per_epoch=traingen.n//traingen.batch_size,
    epochs=1,
    validation_data=valgen,
    validation_steps=valgen.n//valgen.batch_size)
```

    Epoch 1/1
    INFO:tensorflow:New input shapes; (re-)compiling: mode=train, [TensorSpec(shape=(8, 256, 256, 3), dtype=tf.float32, name='input_20'), TensorSpec(shape=(8, 2), dtype=tf.float32, name='dense_1_target_10')]
    INFO:tensorflow:Overriding default placeholder.
    INFO:tensorflow:Remapping placeholder for input
    INFO:tensorflow:Cloning SGD {'lr': 0.009999999776482582, 'momentum': 0.0, 'decay': 0.0, 'nesterov': False}
    INFO:tensorflow:Get updates: Tensor("loss/mul:0", shape=(), dtype=float32)
    INFO:tensorflow:Started compiling
    INFO:tensorflow:Finished compiling. Time elapsed: 28.781575918197632 secs
    INFO:tensorflow:Setting weights on TPU model.
      5/312 [..............................] - ETA: 43:10 - loss: 1.3635 - acc: 0.5250INFO:tensorflow:New input shapes; (re-)compiling: mode=train, [TensorSpec(shape=(4, 256, 256, 3), dtype=tf.float32, name='input_20'), TensorSpec(shape=(4, 2), dtype=tf.float32, name='dense_1_target_10')]
    INFO:tensorflow:Overriding default placeholder.
    INFO:tensorflow:Remapping placeholder for input
    INFO:tensorflow:Cloning SGD {'lr': 0.009999999776482582, 'momentum': 0.0, 'decay': 0.0, 'nesterov': False}
    INFO:tensorflow:Get updates: Tensor("loss_1/mul:0", shape=(), dtype=float32)
    INFO:tensorflow:Started compiling
    INFO:tensorflow:Finished compiling. Time elapsed: 15.506837844848633 secs
    311/312 [============================>.] - ETA: 1s - loss: 0.6510 - acc: 0.6254INFO:tensorflow:New input shapes; (re-)compiling: mode=eval, [TensorSpec(shape=(8, 256, 256, 3), dtype=tf.float32, name='input_20'), TensorSpec(shape=(8, 2), dtype=tf.float32, name='dense_1_target_10')]
    INFO:tensorflow:Overriding default placeholder.
    INFO:tensorflow:Remapping placeholder for input
    INFO:tensorflow:Cloning SGD {'lr': 0.009999999776482582, 'momentum': 0.0, 'decay': 0.0, 'nesterov': False}
    INFO:tensorflow:Started compiling
    INFO:tensorflow:Finished compiling. Time elapsed: 23.947693347930908 secs
    312/312 [==============================] - 495s 2s/step - loss: 0.6512 - acc: 0.6246 - val_loss: 0.6025 - val_acc: 0.6522





    <tensorflow.python.keras.callbacks.History at 0x7f674345db70>



Save the model weights


```
tpu_model.save_weights('./lenet-catdog.h5', overwrite=True)
```

    INFO:tensorflow:Copying TPU weights to the CPU


Download model weights locally


```
from google.colab import files

files.download("lenet-catdog.h5")
```
