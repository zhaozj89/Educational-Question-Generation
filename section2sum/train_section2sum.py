import os
from os.path import join

attribute = 'event'

data_dir = 'bart/{}/data_dir'.format(attribute)
model_name_or_path = 'facebook/bart-base'
tokenizer_name = 'facebook/bart-base'
config_name = 'facebook/bart-base'
gpus=4
max_epochs=3
learning_rate=5e-6
train_batch_size=1
max_target_length=900
val_max_target_length=900
test_max_target_length=900
eval_batch_size=1
cache_dir='pretrained'
output_dir='bart/{}/output_dir'.format(attribute)
path_or_data='data'


cmd_str = 'python {} \
    --data_dir={} \
    --model_name_or_path={} \
    --tokenizer_name={} \
    --config_name={} \
    --do_train \
    --gpus={} \
    --max_epochs={} \
    --learning_rate={} \
    --train_batch_size={} \
    --max_target_length={} \
    --val_max_target_length={} \
    --test_max_target_length={} \
    --eval_batch_size={} \
    --cache_dir={} \
    --output_dir={} \
    --path_or_data={}'.format('1_train.py', 
                              data_dir, 
                              model_name_or_path, 
                              tokenizer_name, 
                              config_name,
                              gpus,
                              max_epochs,
                              learning_rate,
                              train_batch_size,
                              max_target_length,
                              val_max_target_length,
                              test_max_target_length,
                              eval_batch_size,
                              cache_dir,
                              output_dir,
                              path_or_data)

os.system(cmd_str)

