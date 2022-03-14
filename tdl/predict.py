from tqdm import tqdm
from os.path import join
import numpy as np
from pathlib import Path
import math

import joblib
import torch
from torch.nn import functional as F

from train import FairytaleTDL, TYPES, tokenizer, FairytaleQADataset, MAX_TOKEN_COUNT, EPS, TYPE2CONTROL_SIGNAL, NUMBER_CONTROL_SIGNAL

CKT_PATH = 'checkpoints/epoch=2-step=7289.ckpt'

class KL_Loss(torch.nn.Module):
    
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, true, pred):
        true = torch.clamp(true, min=EPS, max=1 - EPS)
        pred = torch.clamp(pred, min=EPS, max=1 - EPS)

        F.normalize(true, p=1, dim=0)
        F.normalize(pred, p=1, dim=0)

        N = true.size()[0]

        # kl loss
        kl = torch.sum(torch.mul(true, torch.log(torch.div(true, pred)))) / N

        return kl

kl_loss = KL_Loss()
 
trained_model = FairytaleTDL.load_from_checkpoint(CKT_PATH, n_classes=len(TYPES) + 1)
                                                  
                                                  
trained_model.eval()
trained_model.freeze()


# input_text = "Hi, I'm Meredith and I'm an alch... good at supplier relations"
# encoding = tokenizer.encode_plus(
#   input_text,
#   add_special_tokens=True,
#   max_length=512,
#   return_token_type_ids=False,
#   padding="max_length",
#   return_attention_mask=True,
#   return_tensors='pt',
# )
# _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
# test_prediction = test_prediction.flatten().numpy()
# for label, prediction in zip(TYPES, test_prediction):
#   print(f"{label}: {prediction}")


data_dir = 'data_dir'
Path(data_dir).mkdir(parents=True, exist_ok=True)

test_data = []
data = joblib.load(join('data', 'data_list.pkl'))['test']
for v in data:
    section = v['input']
    point = {'input': section, 'output': []}
    for item in v['output']:
        question_type = item['question_type']
        
        if question_type not in ['action', 'causal relationship', 'outcome resolution']:
            continue
        else:
            point['output'].append(item)
    if len(point['output'])>0:
          test_data.append(point)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)
test_dataset = FairytaleQADataset(
  test_data,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)
predictions = []
labels = []

section_path = join(data_dir, 'section.txt')
input_section_path = join(data_dir, 'input_section.txt')
gt_question_path = join(data_dir, 'gt_question.txt')
gt_answer_path = join(data_dir, 'gt_answer.txt')
gt_sum_path = join(data_dir, 'gt_sum.txt')
gt_section_path = join(data_dir, 'gt_section.txt')

total_loss = []
# output_data = {}
with open(section_path, 'w') as section_file, open(input_section_path, 'w') as input_section_file, \
  open(gt_question_path, 'w') as gt_question_file, open(gt_sum_path, 'w') as gt_sum_file, \
    open(gt_answer_path, 'w') as gt_answer_file, \
  open(gt_section_path, 'w') as gt_section_file:
  for item in tqdm(test_dataset):
    _, prediction = trained_model(
      item["input_ids"].unsqueeze(dim=0).to(device),
      item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    # predictions.append(prediction.flatten())
    # labels.append(item["labels"].int())
    
    section = item['section']
    questions = item['question']
    answers = item['answer']
    inferences = item['inference']
    gt_sections = item['gt_section']
    prediction = prediction.detach().cpu()
    label = item["labels"].detach().cpu()
    
    section_file.write(section + '\n')
    gt_question_file.write(' [SEP] '.join(questions) + '\n')
    gt_answer_file.write(' [SEP] '.join(answers) + '\n')
    gt_sum_file.write(' [SEP] '.join(inferences) + '\n')
    gt_section_file.write(' [SEP] '.join(gt_sections) + '\n')
    
    loss = kl_loss(label, prediction)
    total_loss.append(loss)
    
    # output_data[section] = questions
    
    mapping = TYPES
    prediction=prediction.tolist()[0]
    modified = False
    section_list = []
    for k, v in enumerate(prediction[:-1]):
          # n = math.ceil(v/prediction[-1]) - 1
          n = int(v/prediction[-1]+0.5)
          if n<=0:
                continue
          for i in range(n):
                modified = True
                section_list.append('{} {} '.format(TYPE2CONTROL_SIGNAL[mapping[k]], NUMBER_CONTROL_SIGNAL[i]) + section)
    if modified==False:
      section_list.append('{} {} '.format(TYPE2CONTROL_SIGNAL[mapping[0]], NUMBER_CONTROL_SIGNAL[0]) + section)
      section_list.append('{} {} '.format(TYPE2CONTROL_SIGNAL[mapping[1]], NUMBER_CONTROL_SIGNAL[0]) + section)
      section_list.append('{} {} '.format(TYPE2CONTROL_SIGNAL[mapping[2]], NUMBER_CONTROL_SIGNAL[0]) + section)
      
    input_section_file.write(' [SEP] '.join(section_list) + '\n')
             
print('mean KL: ', np.mean(total_loss))

# joblib.dump(output_data, 'data_dir/data_eval.pkl')             
    
# predictions = torch.stack(predictions).detach().cpu()
# labels = torch.stack(labels).detach().cpu()