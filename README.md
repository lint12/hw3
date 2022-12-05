# README

Part 1: 
UAS and LAS for your SpaCy parser on en_gum
---------------------------------------------------------------------------
With validation set: {'uas': 0.5725637514477113, 'las': 0.3481523406403405}
With test set: {'uas': 0.5763355074743958, 'las': 0.3502975715335604}

Part 2: 
---------------------------------------------------------------------------
lambda = 0.25: 'rel_pos acc': 0.6714, 'dep_label acc': 0.8527
lambda = 0.5: 'rel_pos acc': 0.7238, 'dep_label acc': 0.8473
lambda = 0.75: 'rel_pos acc': 0.7350, 'dep_label acc': 0.8128

Part 3: 
---------------------------------------------------------------------------
ARGMAX DECODER: 
Results with the validation set
lambda = 0.25: {'uas': 0.7163372462006593, 'las': 0.6829408496078135}
lambda = 0.5: {'uas': 0.7715119337176546, 'las': 0.7303658953167188}
lambda = 0.75: {'uas': 0.781408173124514, 'las': 0.7204365098598424}

Using uas to define the best lambda, the test set result, 
with the best lambda (0.75): {'uas': 0.7951385062488827, 'las': 0.7277712910017539}


MST DECODER: 
Results with the validation set
lambda = 0.25: {'uas': 0.7264734826216458, 'las': 0.6912013968559915}
lambda = 0.5: {'uas': 0.7773746500868033, 'las': 0.7360947437461512}
lambda = 0.75: {'uas': 0.7862979073891726, 'las': 0.7237964463836265}

Using uas to define the best lambda, the test set result, 
with the best lambda (0.75): {'uas': 0.8026289237869635, 'las': 0.731796441177438}
