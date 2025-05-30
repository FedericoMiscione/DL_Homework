# DL_Homework

>**WARNING**  
>The code has been run in a Windows environment. In the file **main.py** has been used an if-elif condition in order to load the model corresponding to the related dataset, but the condition is verified using the *split("\\\\")* function on the string passed as argument.  
>So, if runned in other environments adjust the separator parameter in the *split()* function.

>**WARNING**  
>The dataset has to be uploaded on the directory in order to run the main.py.

>**WARNING**  
>The checkpoints used are named as <model_{dataset_name}_best.pth> 

## Structure of the repository

The repository is organized as follows:
- checkpoints\ : Here are contained all the checkpoints organized for dataset and divided for each model in different folders. The checkpoints used in the ensemble models are located in the subdirectory *ensembled_models*.
- logs\ : Containing logs file for each dataset.
- submission\ : Containing .csv test files.
- source\: Containing the notebook used to run and file related to the implemented or used models.

## Approach

The approach comprises two main ideas:
- Fine tuning of models predefined in the kaggle baseline notebook on all the datasets;
- Implementation of a another module inspired by the GINEConv class of torch_geometric.

Since the results obtained on the fine tuned GIN-based model were satisfactory, in particular on the C and D datasets, the model generalization power has been improved using an ensemble approach on all the datasets. In particular, the ensemble has been composed with a soft voting mechanism that averages the logits produced by a fine tuned model and a fine tuned GINEConv-based model. 

### GIN-based parameters

Fine tuning different hyperparameters of the provided models. In particular, the final version is performed using these parameters:
- gnn_type='gin'
- num_class=6
- num_layer=5
- emb_dim=150
- drop_ratio=0.5
- virtual_node=True
- residual=True
- graph_pooling='attention'

The loss function used is the NoisyCrossEntropyLoss with p_noisy value at 0.5.

The optimizer is Adam with starting learning rate 1e-2 with weight decay 1e-10 with scheduler ReduceLROnPlateau in order to minimize the validation loss.

The model is trained for 50 epochs on the datasets B and D, while 100 epochs on C and 40 epochs on A.
Since a scheduler ReduceLROnPlateau has been applied to the learning rate, the training has been interrupted when it reached the estabilished minimum value for the learning rate [1e-4] and stopped to improve evaluation metrics for a number of epochs equals to the patience of the scheduler.

### GINE-based model

It has been implemented a customized version of a GINE convolutional layer. The reason being the inadequate performance on datasets A and B.
Exploring the *torch_geometric* library we noticed that GINEConv layer is often suggested in graph classification with label noise. In particular, given the obtained results with the GIN-based model and knowing the fact that the GINEConv layer extends the expressive power of traditional GIN by incorporating edge features into the aggregration procedure, we chose to implement a customized version of this convolutional layer in order to integrate it in the GNN model provided through the kaggle baseline notebook.

This model has been trained using a Generalized Cross Entropy loss function with parameter q=0.7 on the dataset B. While for the others the NoisyCrossEntropy has been used with noise_prob parameter set to 0.5.

The optimizer is Adam with starting learning rate 1e-2 with weight decay set to 1e-4, and a ReduceLROnPlateau scheduler has been applied on the following parameters with the following parameters:
- mode = 'max', since the goal was to optimize the f1-score, the step has been applied on the validation f1-score.
- factor = 0.7.
- patience = 4.
- min_lr = 1e-4.

Fine tuning different hyperparameters of the provided models. In particular, the final version is performed using these parameters:
- gnn_type='gine'
- num_class=6
- num_layer=5
- emb_dim=128
- drop_ratio=0.5
- virtual_node=False
- residual=True
- graph_pooling='attention'

### Ensemble

The ensemble has been implemented as a simple wrapper that averages the summed output of the trained models. Since it is a simple wrapper it has not been necessary to train it.
The ensemble improved the generalization power, and so also the performance, obtained with submissions using single models.

In particular, the ensembles are composed by:
- 2 GIN-based and 1 GINEConv-based models trained with different weight decay (model_A_best_1 on 1e-10 and model_best_A_2 on 1e-4) on the dataset A;
- 1 GIN-based model and 1 GINEConv-based model on the dataset B.
- 1 GIN-based model and 1 GINEConv-based model on the dataset C.
- 1 GIN-based model and 1 GINEConv-based model on the dataset D.

## References

https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.GINEConv.html

---

We had to recreate logs of some models on datasets due to some issues with git, since at some point those resulted currupted after the commit on the repository.
The new logs have been regenerated using the metrics contained in the best model's checkpoint for each epoch.
