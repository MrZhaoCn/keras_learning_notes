import tensorflow
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import  Image
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten,Input,LSTM
from keras.utils import plot_model
import  os

data_file = '../data/cmn-tw.txt'

#相关参数
batch_size = 64
epchs = 100
#编码后的维度
latent_dim = 256
num_samples = 10000

input_texts = []
target_texts = []

#英文字符集
input_characters = set()
#中文字符集
tartget_characters = set()

lines = open(data_file, mode='r', encoding='utf-8').read().split('\n')
for line in lines[: min(num_samples, len(lines)-1)]:
    input_text, target_text = line.split('\t')

    #使用tab作为开始序列SOS,使用\n作为结束序列 EOS
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in tartget_characters:
            tartget_characters.add(char)

#全部输入字符集  en
input_characters = sorted(list(input_characters))
#全部目标字符集
tartget_characters = sorted(list(tartget_characters))
# 所有輸入字符的數量
num_encoder_tokens = len(input_characters)
# 所有輸目標字符的數量
num_decoder_tokens = len(tartget_characters)

#最长输入句子长度
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 輸入字符的索引字典
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])

#目标字符索引字典
target_token_index = dict([(char, i) for i, char in enumerate(tartget_characters)])

#包含输入英文句子的one-hot 向量化三维数组 (num_pairs，max_english_sentence_length，num_english_characters）
encode_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens))

# 包含目标中文句子的one-hot向量化的三維形狀數組（num_pairs，max_chinese_sentence_length，num_chinese_characters）
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# decoder_target_data與decoder_input_data相同，但是偏移了一個時間步長。
# decoder_target_data [:, t，：]將與decoder_input_data [：，t + 1，：]相同
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encode_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

#编码
# 注意：因為輸入序列長度(timesteps)可變的情況，使用input_shape =（None，num_features）
encoder_inputs = Input(shape=(None,num_encoder_tokens), name='encode_input')
encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 我們拋棄掉`encoder_outputs`因為我們只需要LSTM cell的內部state參數
encoder_states = [state_h, state_c]

#解码
decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_input')
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='decoder_lastm')

# 在訓練時我們不會使用這些回傳的states, 但是在預測時我們會用到這些states參數
# **解碼器的初始狀態是使用編碼器的最後的狀態(states)**
decoder_outputs,_,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# 接密集層(dense)來進行softmax運算每一個字符可能的機率
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_output')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())

# plot_model(model, to_file='seq2seq_graph.png')
# # Image('seq2seq_graph.png')

#模型训练
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encode_input_data, decoder_input_data],decoder_target_data,batch_size=batch_size, epochs=epchs,validation_split=0.2)
model.save('s2s.h5')

#预测
#定义编码器模型
encoder_model = Model(encoder_inputs, encoder_states)
#定义解码器lstm cell的初始权重输入
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

# 定義解碼器(decoder)的模型
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

# 反向查找字符索引來將序列解碼為可讀的內容。
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1,num_decoder_tokens))
    # 發佈特定的目標序列起始字符"[SOS]",在這個範例中是使用 "\t"字符
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 對符標抽樣
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 停止迴圈的條件: 到達最大的長度或是找到"停止[EOS]"字符,在這個範例中是使用 "\n"字符
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 更新目標序列(of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新 states
        states_value = [h, c]
    return decoded_sentence

for seq_index in range(100):
    # 從訓練集中取出一個序列並試著解碼
    input_seq = encode_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
