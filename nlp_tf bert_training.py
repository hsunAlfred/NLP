import tensorflow as tf
import tensorflow_hub as hub
# 如果遇到奇怪的狀況，嘗試 import tensorflow_text
# import tensorflow_text

x = [
    ['天氣很好'],
    ['天氣好'],
    ['天氣很差'],
    ['天氣差'],
    ['下雨'],
    ['晴天'],
    ['雨'],
    ['晴'],
]
y = [
    3, 2, 0, 1, 1, 2, 1, 2
]

'''
將x y 皆轉換為tensor(張量)
TF變數類型主要有三種
tf.constant：常數，如輸入的x(features) y(label) 
tf.Variable：變數，如weight或bias，可透過學習改善
tf.placeholder：主要是TF在計算Gaph的方法。在使用時，我們是不會先給定一個初始值，而是給定長度或者資料格式先佔有一個空間
'''
tx = tf.constant(x)
# 計算類別數量
nclasses = len(list(set(y)))
# 轉換為類別變數
ty = tf.constant(tf.keras.utils.to_categorical(y, nclasses))

'''
載入預先設定的layer
'''
bert = hub.KerasLayer(
    'https://code.aliyun.com/qhduan/bert_v4/raw/500019068f2c715d4b344c3e2216cef280a7f800/albert_large.tar.gz',
    trainable=True,
    output_key='pooled_output'
)

'''
自訂輸入層
tf.keras.Input(
    shape=None, batch_size=None, name=None, dtype=None, sparse=None, tensor=None,
    ragged=None, type_spec=None, **kwargs
)
shape：A shape tuple (integers), not including the batch size. 
			For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. Elements of this tuple can be None; 
			'None' elements represent dimensions where the shape is not known.
dtype：The data type expected by the input, as a string (float32, float64, int32...)

官方說明文件：https://www.tensorflow.org/api_docs/python/tf/keras/Input
'''
inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.string)

'''
添加輸入層到bert模型
'''
m = bert(inputs)

'''
不確定在幹嘛
'''
# m = tf.keras.layers.Masking()(m)
# m = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, ))(m)

'''
添加輸出層
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
units：Positive integer, dimensionality of the output space.
activation：Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
					relu、softmax、linear

官方說明文件：https://www.tensorflow.org/api_docs/python/tf/keras/Input
'''
outputs = tf.keras.layers.Dense(nclasses, activation='softmax')(m)

'''
建立模型
tf.keras.Model(
    *args, **kwargs
)
inputs：The input(s) of the model: a keras.Input object or list of keras.Input objects.
outputs：The output(s) of the model. See Functional API example below.
name：String, the name of the model.

官方說明文件：https://www.tensorflow.org/api_docs/python/tf/keras/Model
'''
model = tf.keras.Model(inputs=inputs, outputs=outputs)

'''
設定模型訓練方式
compile(
    optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs
)
optimizer：優化方式，Adadelta、Adagrad、Adam、Adamax、FTRL、NAdam、RMSprop、SGD、Optimizer(base)
loss：損失函數，BinaryCrossentropy、CategoricalCrossentropy、CategoricalHinge、CosineSimilarity、Hinge
			Huber、KLDivergence、LogCosh、Loss、MeanAbsoluteError、MeanAbsolutePercentageError
			MeanSquaredError、MeanSquaredLogarithmicError、Poisson、Reduction、SparseCategoricalCrossentropy、SquaredHing

官方說明文件：https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
'''
model.compile(loss='categorical_crossentropy',)

'''
進行訓練
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
x, y：features and label
validation_split：設定驗證資料比例、省略時全部資料作為訓練用
epochs：訓練次數，省略時只訓練1次
batch_size：設定每批次讀取多少資料
verbose：設定是否顯示訓練過程，0不顯示、1詳細顯示、2簡易顯示

官方說明文件：https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
'''
model.fit(tx, ty, epochs=10, batch_size=2)

'''
劃出模型架構圖
需要先安裝
1、graphviz：https://graphviz.gitlab.io/download/
2、python套件：pip install pydot
'''
tf.keras.utils.plot_model(model, to_file='model.png')

'''
predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
'''
logits = model.predict(tx)

'''
The values of logits are Scores, and the indices are Classes. 
By using argmax(), you can obtain the predicted class
'''
pred = logits.argmax(-1).tolist()
