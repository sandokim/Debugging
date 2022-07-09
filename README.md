# Debugging

#### Epoch마다 image plot하여 quality check하는 코드

```python 
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,9.5))

plt.subplot(1, 4, 1)
plt.imshow(image[:,:,150], cmap='CMRmap')
plt.title('input_ori', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(image_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(image_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(image_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 0, top = 0, hspace = 0, wspace = 0)
plt.savefig('./_qc.png')
plt.clf()
plt.close()

del fig
```

```python 
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,9.5))

plt.subplot(1, 4, 1)
plt.imshow(label[:,:,150], cmap='CMRmap')
plt.title('input_ori', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(label_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(label_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(label_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 0, top = 0, hspace = 0, wspace = 0)
plt.savefig('./_qc.png')
plt.clf()
plt.close()

del fig
```

```python 
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,9.5))

plt.subplot(1, 4, 1)
plt.imshow(pseudo_label[:,:,150], cmap='CMRmap')
plt.title('input_ori', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(pseudo_label_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(pseudo_label_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(pseudo_label_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 0, top = 0, hspace = 0, wspace = 0)
plt.savefig('./_qc.png')
plt.clf()
plt.close()

del fig
```

```python
plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 0, top = 0, hspace = 0, wspace = 0)
plt.savefig('./_qc.png')
```

```python 
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,9.5))

plt.subplot(3, 4, 1)
plt.imshow(image[:,:,150], cmap='CMRmap')
plt.title('input_ori', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(image_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(image_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(image_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(label[:,:,150], cmap='CMRmap')
plt.title('label', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(label_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(label_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(label_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(pseudo_label[:,:,150], cmap='CMRmap')
plt.title('pseudo_label', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(pseudo_label_Xshifted[:,:,150], cmap='CMRmap')
plt.title('shifted X', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(pseudo_label_Yshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Y', fontsize=12)
# plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(pseudo_label_Zshifted[:,:,150], cmap='CMRmap')
plt.title('shifted Z', fontsize=12)
# plt.axis('off')

plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 0, top = 0, hspace = 0, wspace = 0)
plt.savefig('./_qc.png')
plt.clf()
plt.close()

del fig
```

### AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'

I got this error when I inadvertently downgraded pytorch to a CPU-only version (by conda installing some other packages).

Should be verifiable via:

```python
>>> torch.cuda.is_available()
False
```
My fix was reinstalling pytorch/torchvision to these pinned versions:

```python
The following packages will be SUPERSEDED by a higher-priority channel:

  pytorch            pkgs/main::pytorch-1.3.1-cpu_py37h62f~ --> pytorch::pytorch-1.3.1-py3.7_cuda10.1.243_cudnn7.6.3_0
  torchvision        pkgs/main::torchvision-0.4.2-cpu_py37~ --> pytorch::torchvision-0.4.2-py37_cu101
```

----

### Computing source check

htop -> RAM check

nvidia-smi -> GPU check

#### 실시간으로 gpu 사용량 확인하기
터미널에서 watch -n0.1 nvidia-smi 를 입력합니다. -> 0.1은 리셋 간극으로 1초를 의미 합니다.

du -h --max-depth=1 -> Current dir storage check

VSCode -> F1 -> interpreter -> python3.9

### Check the number of parameters in Model -> torchsummary

<img src="https://github.com/sandokim/Debugging/blob/main/images/modelsummary.JPG" width="50%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/model_parameters.JPG" width="50%">

#### Github link -> README.md 파일에 적어놓고 시작하기

<img src="https://github.com/sandokim/Debugging/blob/main/images/Readme_file_github_link.PNG" width="100%">

#### F1 -> launch.json -> Debugging전에 args를 준다.

<img src="https://github.com/sandokim/Debugging/blob/main/images/launch_json.PNG" width="100%">

Debugging시 args를 추가하여 모델을 디버깅할 수 있다.

<img src="https://github.com/sandokim/Debugging/blob/main/images/debugging args.PNG" width="100%">

justMycode : False -> import한 package의 연산까지 볼 수 있다.

debug console에서 image input, target input 

<img src="https://github.com/sandokim/Debugging/blob/main/images/debug console.PNG" width="100%">

ex) data.keys()로 argument 확인

ex) dir(batch_data)로 객체확인

ex) batch_data.\__class__\로 batch_data의 class 확인가능 --> numpy array, nibabel 등등

<img src="https://github.com/sandokim/Debugging/blob/main/images/data.keys().PNG" width="80%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/class.PNG" width="60%">

#### debug console로 이미지 확인

import matplotlib.pyplot as plt

plt.imshow(data[0,0,50].cpu(), cmap='gray') # cuda에서 cpu로 옮겨서 plot

plt.savefig('out.png')

#### cmt.txt 파일로 args 설정

<img src="https://github.com/sandokim/Debugging/blob/main/images/cmt파일 및 상대경로설정.PNG" width="100%">

상대경로 설정 : ./~~

keymap 설치하고 F3으로 바로 연결코드 찾기

<img src="https://github.com/sandokim/Debugging/blob/main/images/keymap.PNG" width="50%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/F3.PNG" width="70%">

##### Test의 model.pt가 가진 keys를 디버깅을 통해 확인 --> Debug console -> Model_dict.keys() 확인 --> model_dict['state_dict'].keys()

<img src="https://github.com/sandokim/Debugging/blob/main/images/model_keys_check.PNG" width="70%">

[How to ignore and initialize Missing key(s) in state_dict](https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict/63064444#63064444)

My saved state_dict does not contain all the layers that are in my model. How can I ignore the Missing key(s) in state_dict error and initialize the remaining weights?

<img src="https://github.com/sandokim/Debugging/blob/main/images/model_strict.PNG" width="70%">

##### Debug console 창에서 plt.show, plt.save로 이미지 확인

```python
import matplotlib.pyplot as plt
print(data.shape)
plt.imshow(data[:,:,50], cmap='gray')
plt.savefig('out.png')
```

### input img meta data check

<img src="https://github.com/sandokim/Debugging/blob/main/images/meta_data_check.PNG" width="100%">

### model_dict error --> key 확인 후 state_dict key만 이용하여 모델의 weights와 biases를 불러온다. epoch, best_acc key는 필요없다..!

#### Error가 뜨는 이유는 key값으로 state_dict만 줘야만하기 때문이다..

<img src="https://github.com/sandokim/Debugging/blob/main/images/model_dict.PNG" width="100%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/state_dict.PNG" width="100%">

#### 이로써 아래와 같이 일차적으로 model_dict error가 해결된다.

<img src="https://github.com/sandokim/Debugging/blob/main/images/model_dict['state_dict'].PNG" width="100%">

### CUDA error

[Error] RuntimeError: CUDA error: no kernel image is available for execution on the deviceCUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.

[디바이스에 맞는 torch 설치](https://captainteemo.tistory.com/23)

### [CUDA <-> torch 버전 일치](https://bo-10000.tistory.com/75)

### pip install package error

[ERROR: Command errored out with exit status 1:](https://archivers.tistory.com/669)

[legacy-install-failure(python), 파이썬 버전 낮추면 됨](https://sogogi1000inbun.tistory.com/m/91) --> Doesn't sovle the problem

[setup.py vs requirements.txt](https://jadehan.tistory.com/42)

### [How to state in requirements.txt a direct github source](https://stackoverflow.com/questions/16584552/how-to-state-in-requirements-txt-a-direct-github-source)

#### requirements.txt에 직접 github 소스를 명시하는 방법

<img src="https://github.com/sandokim/Debugging/blob/main/images/github_repository.PNG" width="100%">

### Package install Error

- PackagesNotFoundError

PackagesNotFoundError: The following packages are not available from current channels: conda install에서 가장 흔히 발생하는 오류 중 하나로 conda에서 패키지를 다운로드하려는 기본 채널에 패키지가 존재하지 않는 경우 발생하는데 다음과 같이 해결 가능

[conda install -c conda-forge 패키지명](https://cceeddcc.tistory.com/4)

### Scipy만 설치가 안되는 경우...

pip install -r requiremnents.txt 

requirements.txt에서 scipy만 주석처리하고 설치하면 정상설치 되었지만, 나머지를 모두 주석처리하고 scipy을 설치하는 경우 에러가 발생하였다. Version mismatch?? 

<img src="https://github.com/sandokim/Debugging/blob/main/images/scipy error.PNG" width="80%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/scipy error1.PNG" width="60%">

<img src="https://github.com/sandokim/Debugging/blob/main/images/scipy error2.PNG" width="80%">

이럴때는 그냥 requirements.txt가 아니라 따로 pip install scipy으로 최신버전 scipy를 설치하자.

<img src="https://github.com/sandokim/Debugging/blob/main/images/pip install scipy.PNG" width="80%">

### protobuf error

<img src="https://github.com/sandokim/Debugging/blob/main/images/protobuf.PNG" width="100%">

solution --> pip3 install --upgrade protobuf==3.20.0


