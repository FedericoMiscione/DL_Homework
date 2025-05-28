# DL_Homework

## Idea

Exploring different hyperparameters of the provided models. In particular, the final version is performed using these parameters:
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
Since a scheduler ReduceLROnPlateau has been applied to the learning rate, the training has been interrupted when it reached the estabilished minimum value for the learning rate [1e-6] and stopped to improve evaluation metrics for a number of epochs equals to the patience of the scheduler.
