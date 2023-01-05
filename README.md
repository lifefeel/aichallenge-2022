# 음성 위협상황 인지기술 개발

노이즈 환경에서 일어나는 대화의 내용을 분석하여 위협상황이 존재하는지에 대한 예측을 하는 코드입니다.

## 요구환경

- Ubuntu 18.04 이상 (20.04 권장)
- Python 3.7 이상 (3.8 권장)
- PyTorch 1.7.0 이상
- CUDA 11.2 이상

## 설치

### 코드 다운로드 및 환경 구성

Python 가상환경 설치

```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```

Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 모델 다운로드

모델을 [다운로드](https://sogang365-my.sharepoint.com/:u:/g/personal/jplee_o365_sogang_ac_kr/ETaSZCQAjQhAjYejWBkJd88BFX4an_dkShwcZndxIc6iYA?e=FLWydl) 하셔서 프로젝트 하위폴더에 아래와 같이 압축을 풀어 저장합니다.

```
└─trained_models
  ├─enh_model_sc       # Speech Enhancement Model
  ├─whisper_model      # Speech Recognition Model
  ├─vad_model          # Voice Activity Detection Model
  └─threat_model       # Threat Classifier Model
```

## 데이터

샘플 데이터는 노이즈 상황에서의 사람들간의 대화가 담긴 wav 파일입니다.

```
└─sample_data
  ├─cam1_02.wav       # label: 02051(갈취공갈)
  ├─cam1_05.wav       # label: 000001(일반대화)
  └─cam1_06.wav       # label: 020811(직장내 괴롭힘)
```

### 분류 

위협 상황은 아래와 같이 5가지의 클래스로 분류됩니다. 

- 020121 : 협박
- 000001 : 일반대화
- 02051 : 갈취공갈
- 020811 : 직장내 괴롭힘
- 020819 : 기타 괴롭힘 

## 실행

아래와 같이 실행합니다.

```bash
python main.py
```

### 실행결과

실행을 하면 아래와 같이 대화의 시작과 끝 시간(초), 레이블 예측 결과를 출력합니다.

```
sample_data/cam1_02.wav => 8.0 - 53.75 : 02051
sample_data/cam1_05.wav => 1.75 - 55.0 : 000001
sample_data/cam1_06.wav => 2.0 - 55.25 : 020811
```

## 참고

아래의 모델을 참고하였습니다.

- [ESPnet: Speech Enhancement](https://github.com/espnet/notebook/blob/master/espnet2_tutorial_2021_CMU_11751_18781.ipynb)
- [NeMo Toolkit: VAD Marblenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet) 
- [OpenAI: Whisper](https://github.com/openai/whisper)
- [Transformers: KcELECTRA](https://github.com/Beomi/KcELECTRA)

