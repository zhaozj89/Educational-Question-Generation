### Needed environment

* customized transformers: `cd transformers & pip install .`
* `spacy==2.3.7`
* `torch==1.7.1`
* `pytorch-lightning==0.9.0`
* `torchtext==0.8.0`
* `rouge-score==0.0.4`

### Prepare data

FairytaleQA has not been publically realeased yet, and users need to inquire the authors of FairytaleQA for the dataset. Assuming we have the dataset at `./data/split` and the transformed QA statements at `./data/infrence`, we can prepare the needed format as follows:

* python step1_toxlsx.py
* python step2_topkl.py
* python step3_topkllist.py

### Train and predict question type distribution

In `tdl` folder

* python train.py
* python predict.py

### Train and predict section2sum

In `section2sum` folder

* python train_section2sum.py
* python generate_section2sum.py

### Train and predict sum2question

In `sum2question` folder

* python train_sum2qustion.py
* python generate_sum2question.py