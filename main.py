import pandas as pd
import numpy as np
import tensorflow as tf

from transformers import TFBertModel, TFDistilBertModel
model = TFBertModel.from_pretrained('monologg/kobert', from_pt=True)
from tokenization_kobert import KoBertTokenizer
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일

# 0. HyperParameter ------------------

maxlen = 80
batch_size_HP = 48
epochs_HP = 2
classes = 5

# 1. data preprocessing ----------------

X_data = []
Y_data = []
X_test = []
Y_test = []
train_dataset = pd.read_csv("dataset.csv", encoding='utf-8',sep='|')
test_dataset = pd.read_csv("testset.csv", encoding='utf-8',sep='|')
train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()
k=0

X_data_ids = []
X_data_attn = []
X_data_seg = []
X_test_ids = []
X_test_attn = []
X_test_seg = []

import Preprocess as pr
p = pr.Preprocess(tokenizer,maxlen)

for sentence in train_dataset['Sentence']:
    X_dataset = p.work(sentence)
    X_data_ids.append(X_dataset[0])
    X_data_attn.append(X_dataset[1])
    X_data_seg.append(X_dataset[2])

for sentence in test_dataset['Sentence']:
    X_testset = p.work(sentence)
    X_test_ids.append(X_testset[0])
    X_test_attn.append(X_testset[1])
    X_test_seg.append(X_testset[2])

Y_data = p.labeling(train_dataset, Y_data)
Y_test = p.labeling(test_dataset, Y_test)

X_train_ids = np.array(X_data_ids[:24400])
X_train_attn = np.array(X_data_attn[:24400])
X_train_seg = np.array(X_data_seg[:24400])
X_train = [X_train_ids,X_train_attn,X_train_seg]
Y_train = np.array(Y_data[:24400])

X_val_ids = np.array(X_data_ids[24400:])
X_val_attn = np.array(X_data_attn[24400:])
X_val_seg = np.array(X_data_seg[24400:])
X_val = [X_val_ids,X_val_attn,X_val_seg]
Y_val = np.array(Y_data[24400:])

X_test_ids = np.array(X_test_ids)
X_test_attn = np.array(X_test_attn)
X_test_seg = np.array(X_test_seg)
X_test = [X_test_ids,X_test_attn,X_test_seg]
Y_test = np.array(Y_test)


# 2. modeling --------------------------------

# 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
token_inputs = tf.keras.layers.Input((maxlen,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((maxlen,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((maxlen,), dtype=tf.int32, name='input_segment')
# 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
bert_outputs = bert_outputs[1]
import tensorflow_addons as tfa
# 총 batch size * 4 epoch = 2344 * 4
opt = tfa.optimizers.RectifiedAdam(lr=5.0e-5, total_steps = 2344*4, warmup_proportion=0.1,
                                   min_lr=1e-5, epsilon=1e-08, clipnorm=1.0)
sentiment_drop = tf.keras.layers.Dropout(0.15)(bert_outputs)
sentiment_drop=tf.keras.layers.Dense(35, activation='relu', kernel_initializer=tf.keras.initializers.
                                        TruncatedNormal(stddev=0.02))(sentiment_drop)
sentiment_first = tf.keras.layers.Dense(units=classes, activation='softmax')(sentiment_drop)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
sentiment_model.summary()

sentiment_model.fit(X_train, Y_train, epochs=epochs_HP, batch_size=batch_size_HP, validation_data=(X_val, Y_val))

sentiment_model.save_weights('./kobert_classifier')

print("--------------------model saved!------------------")

sentiment_model.load_weights('./kobert_classifier')
tp = sentiment_model.predict(X_test)
result = []
for temp in tp:
    temp = np.argmax(temp)
    result.append(temp)

k=0
for i in range(len(result)):
    if result[i] == Y_test[i]:
        k+=1

print("정확도는", k/len(Y_test),"입니다")


