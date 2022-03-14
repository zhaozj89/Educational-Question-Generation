import argparse
import json
import time
import re
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import joblib

import argparse
import glob
import logging
import os
from os.path import join
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right


try:
    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from .utils import (
        TYPE2CONTROL_SIGNAL,
        NUMBER_CONTROL_SIGNAL,
        ROUGE_KEYS,
        LegacySeq2SeqDataset,
        Seq2SeqDataset,
        assert_all_frozen,
        calculate_bleu,
        calculate_rouge,
        flatten_list,
        freeze_params,
        get_git_info,
        label_smoothed_nll_loss,
        lmap,
        pickle_save,
        save_git_info,
        save_json,
        use_task_specific_params,
        parse_numeric_cl_kwargs,
    )
except ImportError:
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from utils import (
        TYPE2CONTROL_SIGNAL,
        NUMBER_CONTROL_SIGNAL,
        ROUGE_KEYS,
        LegacySeq2SeqDataset,
        Seq2SeqDataset,
        assert_all_frozen,
        calculate_bleu,
        calculate_rouge,
        flatten_list,
        freeze_params,
        get_git_info,
        label_smoothed_nll_loss,
        lmap,
        pickle_save,
        save_git_info,
        save_json,
        use_task_specific_params,
        parse_numeric_cl_kwargs,
    )


logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
parser.add_argument("--save_path", type=str, help="where to save summaries")
parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
parser.add_argument("--prefix", type=str, required=False, default='', help="will be added to the begininng of src examples")
parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
parser.add_argument("--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all.")
parser.add_argument("--fp16", action="store_true")
################################
parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help='path tooo stored model checkpoints',
    )
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_name_or_path", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
parser.add_argument("--config_name", type=str)
parser.add_argument("--tokenizer_name", type=str)
parser.add_argument("--test_max_target_length", type=int)
parser.add_argument("--eval_max_length", type=int)
################################
# Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
args, rest = parser.parse_known_args()
print(rest)
parsed = parse_numeric_cl_kwargs(rest)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, self.mode)
#         self.target_lens = {
#             "train": self.hparams.max_target_length,
#             "val": self.hparams.val_max_target_length,
#             "test": self.hparams.test_max_target_length,
#         }
#         self.decoder_start_token_id = None  # default to config
#         if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
#             self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
#             self.model.config.decoder_start_token_id = self.decoder_start_token_id
#         self.test_max_target_length = self.hparams.test_max_target_length
   

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)


    def generate(self, input_ids, attention_mask, clause_idx=None, **generate_kwargs):
        # pad_token_id = self.tokenizer.pad_token_id
        # source_ids, source_mask, y = SummarizationDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clause_idx=clause_idx,
            num_beams=1,
            max_length=args.test_max_target_length,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=3,
            early_stopping=True,
            use_cache=False,
            **generate_kwargs
        )
        return generated_ids



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

  
def generate_summaries_examples(
    examples: List[str],
    model,
    device: str = DEFAULT_DEVICE,
    args=None,
    **generate_kwargs,
) -> Dict:
    batch = model.tokenizer(examples, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(device))
    
    summaries = model.generate(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        **generate_kwargs,
    )
    sents = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return sents

def run_generate():
    # examples = [x.split('.')[0]+'.' for x in open(args.input_path).readlines()]

    model: SummarizationModule = SummarizationModule(args)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to('cuda:{}'.format(args.device))
    
    print('#############################################')
    print("# model is loaded from", args.ckpt_path)
    print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    print('#############################################')
    # generate_kwargs['fuse_num']         = args.fuse_num
    # generate_kwargs['type_embedding']   = args.type_embedding

      
    # update config with task specific params
    use_task_specific_params(model, 'summarization')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_file= os.path.join(args.output_dir, 'pred_question.txt')
    
    tgt_lines = []
    input_file = open(args.input_path, 'r').readlines()
    for line in input_file:
        line = line.replace('\n','')
        sections = line.split('[SEP]')
        prefixs = []
        sents = []
        for section in sections:
            prefix = re.search('<[A-Z_]+> <[A-Z]+> ', section).group()
            sent = re.split('<[A-Z_]+> <[A-Z]+> ', section)[1]
            sent = sent.split('.')[0]+'.'
            
            prefixs.append(prefix)
            sents.append(sent)
            
            
        generated_sents = generate_summaries_examples(
            [prefix + sent for (prefix, sent) in zip(prefixs, sents)],
            model,
            device=args.device,
            args=args,
            **parsed,
        )
        
        tgt_lines.append(' [SEP] '.join([prefix + sent for (prefix, sent) in zip(prefixs, generated_sents)]))
            
    fout = Path(result_file).open("w", encoding="utf-8")
    for sent in tgt_lines:
        fout.write(sent + "\n")
    fout.close()   



if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate()
