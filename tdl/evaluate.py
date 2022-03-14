from rouge_score import rouge_scorer
import numpy as np
import joblib
import re
import argparse
import spacy

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, default='../section2sum/bart/event/output/pred_question.txt')
parser.add_argument("--gt_path", type=str, default='data_dir/gt_question.txt')
args = parser.parse_args()

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

precision_1 = []
recall_1 = []
fmeasure_1 = []

precision_2 = []
recall_2 = []
fmeasure_2 = []

precision_L = []
recall_L = []
fmeasure_L = []
with open(args.gt_path, 'r') as gt, open(args.pred_path, 'r') as pred:
    gt_lines = gt.readlines()
    pred_lines = pred.readlines()

    # gt_lines = [line.split('?')[0]+'?' for line in gt_lines]
    # pred_lines = [line.split('?')[0]+'?' for line in pred_lines]
    
    for gt_line, pred_line in zip(gt_lines, pred_lines): 
        gt_sents = gt_line.split('[SEP]')
        pred_sents = pred_line.split('[SEP]')
        gts = []
        preds = []
        for gt_sent in gt_sents:
            gt = re.split('<[A-Z_]+> <[A-Z]+> ', gt_sent)[1]            
            gts.append(gt)

        for pred_sent in pred_sents:
            pred = re.split('<[A-Z_]+> <[A-Z]+> ', pred_sent)[1]
            pred = pred.split('?')[0]+'?'
            pred = pred.replace('\n', '').replace('\r', '')
            if pred not in preds and len(pred.split())>3:
                preds.append(pred)
        gt_line = ' '.join(gts)
        pred_line = ' '.join(preds)    
        
        gt_doc = nlp(gt_line) 
        pred_doc = nlp(pred_line) 
        gt_line = ' '.join([t.text for t in gt_doc])
        pred_line = ' '.join([t.text for t in pred_doc])
        
        score_1 = rouge.score(gt_line, pred_line)['rouge1']
        score_2 = rouge.score(gt_line, pred_line)['rouge2']
        score_L = rouge.score(gt_line, pred_line)['rougeL']

        precision_1.append(score_1.precision)
        recall_1.append(score_1.recall)
        fmeasure_1.append(score_1.fmeasure)

        precision_2.append(score_2.precision)
        recall_2.append(score_2.recall)
        fmeasure_2.append(score_2.fmeasure)
        
        precision_L.append(score_L.precision)
        recall_L.append(score_L.recall)
        fmeasure_L.append(score_L.fmeasure)
        
print('rough 1:')
print('precision: ', np.mean(precision_1))
print('recall: ', np.mean(recall_1))
print('f: ', np.mean(fmeasure_1))

print('*'*10)
print('rough 2:')
print('precision: ', np.mean(precision_2))
print('recall: ', np.mean(recall_2))
print('f: ', np.mean(fmeasure_2))


print('*'*10)
print('rough L:')
print('precision: ', np.mean(precision_L))
# print(np.std(precision_L))
print('recall: ', np.mean(recall_L))
# print(np.std(recall_L))
print('f: ', np.mean(fmeasure_L))
# print(np.std(fmeasure_L))
        
