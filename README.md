# Educational Question Generation of Children Storybooks via Question Type Distribution Learning and Event-centric Summarization

We consider generating high-cognitive-demand (HCD) educational questions by learning question type distribution and event-centric summarization. This repository is the official implementation of [our paper](https://openreview.net/forum?id=QMFQWUBmLDR). 

<div align=center>
<img src="https://github.com/zhaozj89/Educational-Question-Generation/blob/main/images/overview.jpg" width="600">
</div>

## Requirements

Python>=3.6 is needed, run the following commands to install requirements:

```
cd transformers & pip install .
pip install spacy==2.3.7
pip install torch==1.7.1
pip install pytorch-lightning==0.9.0
pip install torchtext==0.8.0
pip install rouge-score==0.0.4
```

## Dataset

FairytaleQA can be found at [here](https://github.com/uci-soe/FairytaleQAData). We used the implementation in `https://github.com/kelvinguu/qanli` to get the QA statements. Thanks to the collaboration, I have the priviliage to use a pre-version of FairytaleQA. But it may not be appropriate for me to share the modified data publicly. If you need a copy of my QA statement data, please write me an email with your name, purpose of use, affliation to zzhaoao@nuist.edu.cn. Thanks very much for your understanding. 

Assuming we have the dataset at `./data/split` and the transformed QA statements at `./data/infrence`, we can prepare the needed format as follows:

```
python step1_toxlsx.py
python step2_topkl.py
python step3_topkllist.py
```

## Training and Prediction

*Paths need to be configured manually*

1. Question type distribution. In `tdl` folder,

```
python train.py
python predict.py
```

2. Event-centric summary generation. In `section2sum` folder,

```
python train_section2sum.py
python generate_section2sum.py
```

3. Educational question generation. In `sum2question` folder, 

```
python train_sum2qustion.py
python generate_sum2question.py
```

## Trained Models

* Question type distribution [here](https://pan.baidu.com/s/1_oH8mSrJgvU2_t8vY-esTg?pwd=324t)

* Event-centric summary generation [here](https://pan.baidu.com/s/19BQlLIW0TzbmbeoYRdtW7Q?pwd=femm)

* Educational question generation [file1](https://pan.baidu.com/s/1yJu9AwZq3voJgA6DonFkeA?pwd=e589) [file2](https://pan.baidu.com/s/1kA2LgGAX1utHAAYaQKkseQ?pwd=e03e), then join them as one file by `join summary2question_epoch=2.ckpt.* > summary2question_epoch=2.ckpt`

## Highlighted Results

* Automatic evaluation on Rouge-L and BERTScore:

<div align=center>
<img src="https://github.com/zhaozj89/Educational-Question-Generation/blob/main/images/automatic.png" width="700">
</div>

* Human evaluation on question types (the K-L distance of question type distribution between our method and groudtruth is 0.28, while QAG (top2) is 0.60):

<div align=center>
<img src="https://github.com/zhaozj89/Educational-Question-Generation/blob/main/images/question_type.png" width="300">
</div>

* Human evaluation on children appropriateness: the mean rating of our method (2.56±1.31) is significantly higher than the one of QAG (top2, 2.22±1.20).

## Acknowledgement

This repository is developed based on [FairytaleQA_QAG_System](https://github.com/WorkInTheDark/FairytaleQA_QAG_System) and [FairytaleQA_Baseline](https://github.com/WorkInTheDark/FairytaleQA_Baseline).

## Citation

```
@inproceedings{zhao2022storybookqag,
    author = {Zhao, Zhenjie and Hou, Yufang and Wang, Dakuo and Yu, Mo and Liu, Chengzhong and Ma, Xiaojuan},
    title = {Educational Question Generation of Children Storybooks via Question Type Distribution Learning and Event-Centric Summarization},
    publisher = {Association for Computational Linguistics},
    year = {2022}
}
```
