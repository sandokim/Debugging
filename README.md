# Debugging

Debugging시 args를 추가하여 모델을 디버깅할 수 있다.

<img src="https://github.com/sandokim/Debugging/blob/main/images/debugging args.PNG" width="60%">

justMycode : False -> import한 package의 연산까지 볼 수 있다.

debug console에서 image input, target input 

<img src="https://github.com/sandokim/Debugging/blob/main/images/debug console.PNG" width="100%">

ex) data.keys()로 argument 확인

ex) dir(batch_data)로 객체확인

ex) batch_data.\__class__\()로 batch_data의 class 확인가능

<img src="https://github.com/sandokim/Debugging/blob/main/images/data.keys().PNG" width="80%">

#### debug console로 이미지 확인

import matplotlib.pyplot as plt

plt.imshow(data[0,0,50].cpu(), cmap='gray') # cuda에서 cpu로 옮겨서 plot

plt.savefig('out.png')

#### cmt.txt 파일로 args 설정

<img src="https://github.com/sandokim/Debugging/blob/main/images/cmt파일 및 상대경로설정.PNG" width="100%">

상대경로 설정 : ./~~
