import joblib
from pathlib import Path
import pandas as pd
import neuralcoref
import spacy



nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp, greedyness=0.45)

counter = 0
counter_q = 0

data = {}
for split in ['train', 'val', 'test']:
    mapping_path='data/{}.xlsx'.format(split)
    mapping = pd.read_excel(mapping_path)
    
    data[split] = {}
    
    unique_section = []
    for i in range(mapping.shape[0]):
        if mapping.loc[i, 'local-or-sum']=='summary' or not isinstance(mapping.loc[i, 'inference'], str):
            continue
        section = mapping.loc[i, 'cor_section']
        question = mapping.loc[i, 'question']
        inference = mapping.loc[i, 'inference']
        answer = mapping.loc[i, 'answer1']
        question_type = mapping.loc[i, 'attribute1']
        
        book_id = mapping.loc[i, 'book_id']
        book_name = mapping.loc[i, 'book_name']
        section_id = mapping.loc[i, 'section_id']
        
        section = section.replace('\n', ' ').replace('\r', ' ')
        question = question.replace('\n', ' ').replace('\r', ' ')
        inference = inference.replace('\n', ' ').replace('\r', ' ')
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        
        # doc = nlp(section)._.coref_resolved
        # doc = nlp(doc)
        # sents = [c.string.strip() for c in doc.sents]
        # section = ' '.join(sents)
        
        if section not in unique_section:
            unique_section.append(section)
            data[split][section] = []
            
        data[split][section].append({'question': question, 'type': question_type, 'inference': inference, 'answer': answer, \
            'book_id': book_id, 'book_name': book_name, 'section_id': section_id})
        counter_q +=1
    print(len(unique_section)) 
    print(counter_q)        
    counter+=len(unique_section)
print(counter)
print(counter_q)

joblib.dump(data, 'data/data.pkl')