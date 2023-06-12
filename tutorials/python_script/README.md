# Deep learning Python Script

In this tutorial, we learn about:
* [Python modules](https://docs.python.org/3/tutorial/modules.html)
* [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)
* Running a script on the server

We use the same code as we studied in our last session (Notebook:
[Probing with Linear Classifiers](notebooks/linear_classifier_probe.ipynb)). We convert the code
into a python module named *deepcsf*.

To execute this code, first we have to activate our virtual environment containing necessary
packages like PyTorch (check the [environment setup tutorial](../environment_setup.md)).


Assuming you are already in the *server* directory where the *deepcsf* module is, to train a
network:

    python main.py

And to test the trained network:

    python main.py --test_net <CHECKPOINT_PATH>

The `CHECKPOINT_PATH` is the path to saved checkpoint in the training script, by default it's saved
at `csf_out/checkpoint.pth.tar`.

# Python Module

The `main` function receives several arguments that we cover during this tutorial.

    usage: main.py [-h] [--epochs EPOCHS] [--initial_epoch INITIAL_EPOCH] [--batch_size BATCH_SIZE]
                   [--train_samples TRAIN_SAMPLES] [--num_workers NUM_WORKERS] [--lr LR] 
                   [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--out_dir OUT_DIR] 
                   [--test_net TEST_NET] [--resume RESUME]
    
    options:
      -h, --help            show this help message and exit
      --epochs EPOCHS       number of epochs of training
      --initial_epoch INITIAL_EPOCH
                            the staring epoch
      --batch_size BATCH_SIZE
                            size of the batches
      --train_samples TRAIN_SAMPLES
                            Number of train samples at each epoch
      --num_workers NUM_WORKERS
                            Number of CPU workers
      --lr LR               SGD: learning rate
      --momentum MOMENTUM   SGD: momentum
      --weight_decay WEIGHT_DECAY
                            SGD: weight decay
      --out_dir OUT_DIR     the output directory
      --test_net TEST_NET   the path to test network
      --resume RESUME       the path to training checkpoint

