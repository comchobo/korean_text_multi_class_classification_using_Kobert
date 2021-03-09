import re
import numpy as np
class Preprocess():
    def __init__(self, tokenizer,maxlen):
        self.tk = tokenizer
        self.maxlen = maxlen

    def work(self, sentence):
        sentence = re.sub('\,|\"|=|<|>|\*|\'', '', sentence)
        sentence = re.sub('\(|\)', ',', sentence)
        sentence = re.sub('[0-9]+', 'num', sentence)
        #sentence = re.sub(";+", ';', sentence)
        #sentence = re.sub("[?]{2,}", '??', sentence)
        sentence = re.sub("[.]{2,}", '..', sentence)
        #sentence = re.sub("[!]{2,}", '!!', sentence)
        #sentence = re.sub('~', '', sentence)
        #sentence = re.sub('[a-zA-Z]', '', sentence)

        token = self.tk.encode(sentence, max_length=self.maxlen, pad_to_max_length=True)

        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(1)
        mask = [1] * (self.maxlen - num_zeros) + [0] * num_zeros

        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0] * self.maxlen

        #token = np.array(token)
        #mask = np.array(mask)
        #segment = np.array(segment)

        return [token, mask, segment]

    def labeling(self, data, train):
        for i in range(len(data['Emotion'])):
            if data['Emotion'].iloc[i] == '슬픔':
                train.append([0])
            elif data['Emotion'].iloc[i] == '중립':
                train.append([1])
            elif data['Emotion'].iloc[i] == '행복':
                train.append([2])
            elif data['Emotion'].iloc[i] == '공포':
                train.append([3])
            elif data['Emotion'].iloc[i] == '분노':
                train.append([4])
        return train
