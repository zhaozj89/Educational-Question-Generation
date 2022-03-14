import os
from os.path import join


attribute = 'event'

model_name_or_path = 'facebook/bart-large'
tokenizer_name = 'facebook/bart-large'
config_name = 'facebook/bart-large'
ckpt_path='bart/{}/output_dir/epoch=2.ckpt'.format(attribute)
input_path='/home/amax/zzhaoao/BookComprehension/FinetuneBART/section2sum/bart/{}/output/pred_sum.txt'.format(attribute)
bs=8
device=1
cache_dir='pretrained'
output_dir='/home/amax/zzhaoao/BookComprehension/FinetuneBART/section2sum/bart/{}/output/'.format(attribute)
test_max_target_length=20

cmd_str = 'python {} \
    --model_name_or_path={} \
    --tokenizer_name={} \
    --config_name={} \
    --ckpt_path={} \
    --input_path={} \
    --bs={} \
    --device={} \
    --cache_dir={} \
    --output_dir={} \
    --test_max_target_length={}'.format('2_generate.py',  
                                        model_name_or_path, 
                                        tokenizer_name, 
                                        config_name,
                                        ckpt_path,
                                        input_path,
                                        bs,
                                        device,
                                        cache_dir,
                                        output_dir,
                                        test_max_target_length)

os.system(cmd_str)