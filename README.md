# text-emotion-recognition-6-class

## About 
Predicting the emotion from text utterances. 6 possible emotion labels. IEMOCAP dataset used.

## Instructions for running

1. pip install requirements.txt
2. python create_dialogues.py --num_classes=<NUM_CLASSES> --past=<PAST>
      (To create dialogue buckets with 'past' number of previous utterances considered in making the dialogue. Stores all the bucketed messages in pickle file in 'data/data-<NUM_CLASSES>-<PAST>/' directory.
3. python main.py --num_classes=<NUM_CLASSES> --past=<PAST>
      (Train and test the model. With dataset of NUM_CLASSES and PAST utterances considered. 
      NOTE: If the dataset is not created first, then it won't be able to run this file, so first create the dataset of your desired classes and past.)
      
## Example

1. python create_dialogues.py --num_classes=6 --past=2

( data/data-6-2/ : Directory which stores pickle files of bucketed data)

2. python create_dialogues.py --num_classes=6 --past=2 -lr_roberta= 2e-6
(Train model with learning rate of RoBERTa set to 2e-6)
