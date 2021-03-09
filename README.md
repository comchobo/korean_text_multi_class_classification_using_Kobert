# korean_text_multi_class_classification_using_Kobert<br>

used Tensorflow<br>

bert는 monologg님의 kobert를 사용했습니다.

https://github.com/monologg/KoBERT-Transformers 참고

sentiment multi-class classify하는 모델입니다. 정확도는 multilingual-cased에 비해 7%가량 향상됐습니다 (62->68%)

dataset.csv과 testset.csv는 Aihub의 데이터 (https://aihub.or.kr/node/274) 를 사용하였습니다.
<br>
데이터 예시:<br>
Sentence|Emotion<br>
난 원피스고 나왔으면 그러면 속초간다 연차내고|중립<br>
방탈죄송합니다ㅠ|슬픔<br>
...<br>
<br>
https://parksrazor.tistory.com/231 이곳의 코드를 참고하였습니다. 훈련시간은 1080기준 15분가량입니다.
