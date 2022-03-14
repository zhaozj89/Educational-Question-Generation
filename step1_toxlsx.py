import os
import pandas as pd
from pathlib import Path

Path('data').mkdir(parents=True, exist_ok=True)

inference_path = './data/inference/'

def check_questionfile(file):
    return (len(file)-len('questions')-4)==file.rfind('questions')

def read_lines(filename):
    lines = open(filename, 'r').readlines()
    return [line.replace('\n', '') for line in lines if line.replace('\n', '')!='']

def main():
    q_counter = 0
    
    book_id = 0
    for split in ['train', 'val', 'test']:
        mapping = None
        
        DATA_PATH = './data/split/{}'.format(split)
        for subdir, dirs, files in os.walk(DATA_PATH):
            for file in files:
                if check_questionfile(file):
                    q_df = pd.read_csv(os.path.join(subdir, file))
                    s_df = pd.read_csv(os.path.join(subdir, file.replace('questions', 'story')))
                    
                    col_names=q_df.columns.tolist()  
                    col_names.insert(2,'inference') 
                    q_df=q_df.reindex(columns=col_names)
                    
                    col_names=q_df.columns.tolist()  
                    col_names.insert(2,'book_id') 
                    q_df=q_df.reindex(columns=col_names)
                    
                    col_names=q_df.columns.tolist()  
                    col_names.insert(2,'book_name') 
                    q_df=q_df.reindex(columns=col_names)
                        
                    col_names=q_df.columns.tolist()  
                    col_names.insert(2,'section_id') 
                    q_df=q_df.reindex(columns=col_names)
                    
                    counter = {}
                    for i in range(q_df.shape[0]):
                        if q_df.loc[i, 'local-or-sum']=='summary' or ',' in str(q_df.loc[i, 'cor_section']):
                            continue
                        idx = q_df.loc[i, 'cor_section']
                        
                        inference_name = inference_path + split + '_' + file.replace('.csv', '_') + str(int(float(idx))) + '.txt'
                        inference_lines = read_lines(inference_name)
                                            
                        if idx in counter:
                            counter[idx] += 1
                        else:
                            counter[idx] = 0

                        # print(counter[idx])
                        # print(len(inference_lines))
                        print(subdir, file, idx)
                        inference = inference_lines[counter[idx]]
                        q_df.loc[i, 'inference'] = inference
                        
                        is_modify = False
                        for k, j in enumerate(s_df['section'].to_list()):
                            if int(float(j))==int(float(idx)):
                                q_df.loc[i, 'cor_section'] = s_df.loc[k, 'text'].replace('\n',' ').replace('\r',' ')
                                q_df.loc[i, 'book_id'] = book_id
                                q_df.loc[i, 'book_name'] = file.replace('questions', 'story')
                                q_df.loc[i, 'section_id'] = idx
                                is_modify = True
                                break
                        if is_modify == False:
                            raise ValueError('error')
                        
                    if mapping is None:
                        mapping = q_df
                    else:
                        mapping = pd.concat([mapping, q_df])     
                    q_counter+=q_df.shape[0]       
                book_id+=1
        print(mapping.shape[0])
        mapping.to_excel('data/{}.xlsx'.format(split))
    print(q_counter)
    
main()
