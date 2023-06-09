LFCC-LCNN baseline for ASVspoof 2021

General
----------
baseline_LA: model and configuration files for LA baseline
baseline_PA: model and configuration files for PA baseline
baseline_DF: model and configuration files for DF baseline

Usage (with a toy data set as example)
--------------------------------------
1. Download pre-trained model and toy dataset
   $: bash 00_download.sh

2. If you simply want to evaluate trials using the pre-trained model
   $: bash 01_wrapper_eval.sh

   This first script shows the conveninent way to evaluate trials in one folder.
   You just need to specify the path to the folder, the name of this trial set (any string),
   and the path to the trained model.

3. If you want to re-train the model and do evaluation 
   $: bash 02_toy_example.sh

   Walking through the second script shows how to use this project to train and evaluate
   a data set. It is useful if you want to train a model using other data sets.

   When using other datasets, the dataset directory and related configuration files must 
   be prepared in advance. After running 02_toy_example.sh, you will see how the data set 
   is prepared in DATA/toy_example. And you can check how configuration is done in */config.py

Note that running the script for the first time will install conda dependency, which may
take some time.

Als note that the toy dataset is a tiny part of the ASVspoof2019 LA database, and it is 
used as toy dataset for all the three baselines baseline_LA, baseline_PA, baseline_DF.

Pre-trained models in baseline_LA and _DF were trained using ASVspoof2019 LA train set.
That in baseline_PA was trained using ASVspoof2019 PA train set.


Folder structure
----------------
Files marked with '=' are generated after running 02_toy_example.sh.

|- DATA: data directory for the toy dataset
|  |= toy_example.tar.gz: toy dataset package
|  |= toy_example
|  |  |= protocol.txt: 
|  |  |   protocol file
|  |  |   this will be loaded by pytorch code for training and evaluation
|  |  |= scp: 
|  |  |   list of files for traing, dev, and eval
|  |  |= train_dev: 
|  |  |   waveform for train and validation sets   
|  |  |= eval: 
|  |  |   waveform for evaluation sets
|
|- baseline_LA: folder for baseline LA
|  |  
|  |  |- 00_train.sh: recipe to train the model
|  |  |- 01_eval.sh: command line to evaluate the model on eval set
|  |  |- 02_eval_alternative.sh: convenient script used by 01_wrapper_eval.sh
|  |  |- main.py: main function
|  |  |- config.py: configuration file
|  |  |- config_auto.py: configuration file used by 02_eval_alternative.sh
|  |  |- model.py: definition of model in Pytorch code
|  |  |- __pretrained
|  |  |    |- trained_network.pt: pre-trained model
|  |  |
|  |  |= epochNNN.pt: trained model after NNN-th epoch
|  |  |= trained_network.pt: trained model after the full training process
|  |  |= asvspoof2021_*_toy_utt_length.dic: 
|  |  |    cache files to save the duration information of each trial
|  |  |    they are automatically generated by the code
|  |  |    they can be deleted freely
|  |  |
|  |  |= log_train.txt: training log
|  |  |= log_err.txt: error log (it also contains the training loss of each trial)
|  |  |= log_eval.txt: evaluation log
|  |  |= log_eval_score.txt: score of each trial, extracted from log_eval.txt
|  |  |= log_eval_err.txt: error log 

log_train.txt is self-explanable.

log_eval_score.txt has two columns (see Note 2):
  File_name score
  LA_E_1066571 19.160793


Note
----------
1. If GPU memory is insufficient, please reduce --batch-size in */00_train.sh

2. Output score has this following format:

   File_name score
   LA_E_1066571 19.160793
   
   The score is a scalar indicating how likely the trial is being bona fide (human voice).
   A higher score means being more likely to be bona fide.

3. Accordingly, the code assumes 0 and 1 as the target labels for spoof and bona fide
   trials, respectively. 

4. Although you can use an arbiary string to name the data set in config.py and
   02_eval_alternative.sh, it is better to use a meaningful name and differentiate 
   the names for different data sets. 
   For example, we use asvspoof2021_train_toy, asvspoof2021_val_toy, asvspoof2021_eval_toy
   in config.py.
     
5. The used countermeasure model is detailed here https://arxiv.org/abs/2103.11326.
   However, the LFCC front-end uses a slightly different configuration: the maximum frequency
   of the linear filter bank was set to 4kHz, not 8kHz.

That's all