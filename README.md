# README

Part 1: 
---------------------------------------------------------------------------
UAS and LAS for SpaCy parser on en_gum
- With validation set: 
  - 'uas': 0.5725
  - 'las': 0.3481
- With test set: 
  - 'uas': 0.5763
  - 'las': 0.3502

Part 2: 
---------------------------------------------------------------------------
- *** finetune_bert.py is for preprocessing and creating the vocab set 
- *** finetune_bert.ipynb is for actual finetuning, training, and evaluations 
- lambda = 0.25: 
  - 'rel_pos acc': 0.6714
  - 'dep_label acc': 0.8527
- lambda = 0.5: 
  - 'rel_pos acc': 0.7238
  - 'dep_label acc': 0.8473
- lambda = 0.75: 
  - 'rel_pos acc': 0.7350
  - 'dep_label acc': 0.8128

Part 3: 
---------------------------------------------------------------------------
#### ARGMAX DECODER: 
- Results with the validation set
  - lambda = 0.25: 
    - 'uas': 0.7163
    - 'las': 0.6829
  - lambda = 0.5: 
    - 'uas': 0.7715
    - 'las': 0.7303
  - lambda = 0.75: 
    - 'uas': 0.7814
    - 'las': 0.7204

- Using 'uas' to define the best lambda, the test set result, with the best lambda (0.75): 
  - 'uas': 0.7951
  - 'las': 0.7277


#### MST DECODER: 
- Results with the validation set
  - lambda = 0.25: 
    - 'uas': 0.7264
    - 'las': 0.6912
  - lambda = 0.5: 
    - 'uas': 0.7773
    - 'las': 0.7360
  - lambda = 0.75: 
    - 'uas': 0.7862
    - 'las': 0.7237

- Using 'uas' to define the best lambda, the test set result, with the best lambda (0.75): 
  - 'uas': 0.8026
  - 'las': 0.7317
