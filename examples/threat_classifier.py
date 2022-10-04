import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
#from fastprogress.fastprogress import master_bar, progress_bar
#from attrdict import AttrDict
import numpy as np
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    ElectraTokenizer,
    ElectraForSequenceClassification
)

tokenizer = ElectraTokenizer.from_pretrained('/root/sogang_asr/threat_model/baseline-kcelectra-newnew_train/tokenizer')
model = ElectraForSequenceClassification.from_pretrained('/root/sogang_asr/threat_model/baseline-kcelectra-newnew_train/epoch-26')# 모델 경로 넣기

f = open("/root/sogang_asr/threat_model/newnew_sample_100.txt", 'r',encoding='utf8')# 테스트 파일 넣기
lines = f.readlines()
reallabel=[]
content=[]
for line in lines:
    asr,label = line.split('\t')
    content.append(asr.strip())
    reallabel.append(label.strip())
f.close()

predlist=[]
for i in content:
    inputs=tokenizer(i, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    pred=model.config.id2label[predicted_class_id]
    predlist.append(pred)

predlist=[str(p) for p in predlist]

from sklearn.metrics import accuracy_score
print(accuracy_score(reallabel,predlist))