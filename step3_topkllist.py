import joblib
from os.path import join


data_list = {}
for split in ['train', 'val', 'test']:
    data = joblib.load(join('data', 'data.pkl'))[split]
    
    split_list = []
    for k,v in data.items():
        type_counter = {}
        section = k
        point = {'input': {'section': section}, 'output': []}
        for item in v:
            question = item['question']
            inference = item['inference']
            answer = item['answer']
            question_type = item['type']
            book_id = item['book_id']
            book_name = item['book_name']
            section_id = item['section_id']
            
            point['input']['book_id'] = book_id
            point['input']['book_name'] = book_name
            
            point['output'].append({'question': question, 'inference': inference, \
                'question_type': question_type, 'answer': answer, 'section_id': section_id})
        split_list.append(point)
    data_list[split] = split_list
    

joblib.dump(data_list, 'data/data_list.pkl')          
            
            
        