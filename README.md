## Context-Dependent Sentiment Analysis in User-Generated Videos

### Requirements
Code is written in Python (3.5) with Jupter Notebook and requires Pytorch (0.4.0) for training. GPU is not optional for model training.

### Description
We follow the work below to replicate the result using LSTM-based model that enables utterances to capture contextual information.
```
@inproceedings{soujanyaacl17,
  title={Context-dependent sentiment analysis in user-generated videos},
  author={Poria, Soujanya  and Cambria, Erik and Hazarika, Devamanyu and Mazumder, Navonil and Zadeh, Amir and Morency, Louis-Philippe},
  booktitle={Association for Computational Linguistics},
  year={2017}
}
```

### Dataset
We provide results on the [MOSI dataset](https://arxiv.org/pdf/1606.06259.pdf)  

### Preprocessing

We used the same code provided by the author for data processing.
The code is: 

```
python create_data.py
```

Note: This will create speaker independent train and test splits 

### Running Model

All training are done in Multimodel.ipynb
Simply follow the notebook and run the following codes for unimodel training 

```
train_unimodel(epochs = 30, 
               batch_size = 10,
      validation_split = 0.2, 
      stop_early = 15, 
      hidden_size = 300, 
      dropout = 0.9)
```
Note: stop_early should be optional.
After training, 100 dimension features are generated for each modality and saved as "result/mode_unimodel_epoch_30.pickle"
Results (Training, validation, and test accuracy and loss) are saved in "result/mode_unimodel_epoch_30.pickle".

For multimodel training, run the following code:
```
train_multimodel(epochs = 50, stop_early = 20)
```

### TODO

Current multimodal training does not yield further improvement over unimodel training. Next step is to change fusion strategy and double check training strategy to improve results.

### Developers

#### Tianyi Wu
M.S. student, University of California, San Diego
email: tiw206@ucsd.edu  
 




