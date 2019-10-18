# asr_hackathon
naver ai asr hackathon 2019

기존 SOTA
1. Mel40 VGG-CNN Style, dec_layer=4, d_model=512, lr=1e-7
2. Spectrogram with same weights

바로 해야할 것
1. Dataset
```
  1-1) 전체 셋 50k 중, 30k train, 20k test ==> 재석
  1-2) batch_size=20일 때, original_set / mask1/1_set/ mask2/1_set 으로 input 구현 ==> 재석
  1-3) Data Augmentation with SpecAugment ==> 준우, 코드 완료
```
2. Preprocessing
```
  2-1) Normalize --> local에서 학습할때에는 30k 에 대해서 min,max find하여 이걸로 normalization함 (train 에만 normalization 진행)
    2-1-1) Denormalization 할 필요는 없음
  2-2) log --> np.log10 을 사용하여 전체 값을 줄
  2-3) zero padding -> 이미 코드내에 구현되어 있음
  
Summary np.log10 --> Normalization --> Zero padding
```

TO DO
```
1. mel_dim = 몇으로 할 것인가?
2. Encoder 에서, 기존 잘 나오는 모델의 VGG-Style CNN을 쓸 것인가? 아니면 3DCNN + BLSTM 쓸 것인가?
2-1) 준현이의 모델은 시간이 오래걸리지만 loss가 안튀는 장점이 있다. 하지만 해커톤 하루 동안 새롭게 돌리는것은 불가능 할 것이다.
2-2) 기존 우리의 SOTA였던 mel40 VGG-Style로 가게 되면, loss가 튀긴 하지만, 빠른 시간내에 학습이 가능한(해커톤에서 하루만에 학습 가능) 이점이 있다...
2-3) 정호영 교수님이 추천하신 모델은 valid cer이 10% 미만으로 떨어지지 않긴 하지만, 튀지 않는 다는 장점이 있다.
3. Decoder에서 3개 정도의 다른 mask를 혹은 condition을 이용한 Ensemble?
==> 월요일에 정호영 교수님께 Ensemble 질문할것 
```
