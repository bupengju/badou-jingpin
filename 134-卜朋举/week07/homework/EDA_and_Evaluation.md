# EDA

total samples: positive: 4000; negative: 7987

data split method: random split

train ratio: 0.8

train dataset: positive: 3183; negative: 6386

test dataset: positive: 817; negative: 1601

# Evaluation



|    mdoel     | epoch | lr    | batch size | hidden size |  acc   | time (eval 100 sample) |
| :----------: | :---: | ----- | :--------: | :---------: | :----: | :--------------------: |
|   fasttext   |  20   | 0.001 |     32     |     64      | 0.8756 |         0.17 s         |
|   textrnn    |  20   | 0.001 |     32     |     64      | 0.8655 |         0.49 s         |
|     rnn      |  20   | 0.001 |     32     |     64      | 0.8551 |         0.26 s         |
|     lstm     |  20   | 0.001 |     32     |     64      | 0.8693 |         0.34 s         |
|   textcnn    |  20   | 0.001 |     32     |     64      | 0.8723 |         0.16 s         |
|   gatedcnn   |  20   | 0.001 |     32     |     64      | 0.8750 |         0.33 s         |
|     bert     |  20   | 0.001 |     32     |     64      |   -    |           -            |
|   bertlstm   |  20   | 0.001 |     32     |     64      |   -    |           -            |
|   bertcnn    |  20   | 0.001 |     32     |     64      |   -    |           -            |
| bertmidlayer |  20   | 0.001 |     32     |     64      |   -    |           -            |

