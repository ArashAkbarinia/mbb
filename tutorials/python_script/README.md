# Deep learning Python Script

In this tutorial, we learn about:
* [Python modules](https://docs.python.org/3/tutorial/modules.html)
* [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)
* Running a script on the server

We use the same code as we studied in our last session (Notebook:
[Probing with Linear Classifiers](../../notebooks/linear_classifier_probe.ipynb)). We convert the 
code into a Python package named `deepcsf`.

To execute this code, first, we have to activate our virtual environment containing necessary
packages like PyTorch (check the [environment setup tutorial](../environment_setup.md)).


Assuming you are already in the *server* directory where the `deepcsf` module is, to train a
network:

    python main.py

And to test the trained network:

    python main.py --test_net <CHECKPOINT_PATH>

The `CHECKPOINT_PATH` is the path to the saved checkpoint in the training script, by default, it's saved
at `csf_out/checkpoint.pth.tar`.

***

# Python Module

Jupyter Notebook provides an interactive programming environment. This is very useful in several 
scenarios such as:
* prototyping ideas
* exploring data
* plotting results
* demo codes
* etc.

However, training real-world deep networks often consist of a larger magnitude of code which
is difficult to manage in Notebooks. To this end, we should create Python modules and scripts:
* **Python script**: is an executable file that can be executed in the terminal, e.g., 
```python <SCRIPT_PATH>.py```.
* **Python module**: contains function definitions similar to a third-party library or a package.

In this tutorial, we have created a minimal Python package called `deepcsf` following this 
structure:
```
python_script/
└── deepcsf/                # Python package
    └── __init__.py         # __init__.py is required to import the directory as a package
    └── csf_main.py         # training/testing routines
    └── dataloader.py       # dataset-related code
    └── models.py           # the architecture of the network
    └── utils.py            # common utility functions
└── main.py                 # executable script
```

**NOTE**: This tutorial contains a single Python package and a single script, a more complex project 
often contains several packages and scripts.

## Arguments

The [argparse](https://docs.python.org/3/library/argparse.html) module makes it easy to write 
user-friendly command-line interfaces. Our `main.py` module receives several arguments. We can see
the list of arguments by calling:

    python main.py --help

Which outputs:

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

***

# TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard/) provides the visualisation and tooling needed for machine learning experimentation:
* Tracking and visualising metrics such as loss and accuracy
* Visualising the model graph (ops and layers)
* Viewing histograms of weights, biases, or other tensors as they change over time
* Projecting embeddings to a lower dimensional space
* Displaying images, text, and audio data.

## Logging

In the [csf_main.py](deepcsf/csf_main.py) we have used TensorBoard to:
* log accuracy and loss values
* show batch images

The `SummaryWriter` class is your main entry to log data for consumption and visualisation by
TensorBoard. So, we import it:

    from torch.utils.tensorboard import SummaryWriter

At the start, we initialise two instances of `SummaryWriter` for train and testing, each logging in
their corresponding directories:

    args.tb_writers = dict()
    for mode in ['train', 'test']:
        args.tb_writers[mode] = SummaryWriter(os.path.join(args.out_dir, mode))

We add new accuracy/loss by calling the `add_scalar` function and add new images by calling the 
`add_image` function.

<img src="https://pytorch.org/docs/stable/_images/add_scalar.png"  width="400" height="200">

<img src="https://pytorch.org/docs/stable/_images/add_image.png"  width="400" height="200">

`SummaryWriter` contains several `add_<SOMETHING>` functions 
(https://pytorch.org/docs/stable/tensorboard.html), most of them with a similar set of arguments:
* tag (data identifier)
* value (e.g., a floating number in case of scalar and a tensor in case of image)
* step (allowing to browse the same tag at different time steps)

At the end of the programme, it's recommended to close the `SummaryWriter` by calling the `close()`
function.

## Monitoring

We can open the TensorBoard in our browser by calling

    tensorboard --logdir <LOG_DIR> --port <PORT_NUMBER>

In our project, by default, the TensorBoard files are saved at *csf_out/train/* and *csf_out/test/*
folder. If we specify the `<LOG_DIR>` as the parent directory (*csf_out/*), TensorBoards in all
subdirectories will be also visualised:
* This is a very useful tool to compare different conditions (e.g., train/test, 
different experiments) at the same time.
* If there are too many nested TensorBoards, it might become too slow.

The value for `<PORT_NUMBER>` is a four-digit number, e.g., 6006.:
* If the port number is already occupied by another process, use another number.
* You can have several TensorBoards open at different ports.

Finally, we can see the TensorBoard in our browser under this URL

    http://localhost:<PORT_NUMBER>/

***

# Running on the server

In a real deep learning project, we often need to run our code on a server computer with powerful 
GPUs.

## Connection

We can use the Secure Shell Protocol (**SSH**) to connect to a server computer:

    ssh <USER_NAME>@<SERVER_IP>

After entering the password, you have made a connection to the server computer. You have access to
the terminal of the server computer where you can execute different commands.

## Jupyter

You can start a `jupyter notebook/lab` on the server and open it on your local browser.

On the server run:

    jupyter notebook --no-browser --port=<REMOTE_PORT>

This is similar to how we start a `jupyter notebook/lab` on local machine with an extra argument of
`--no-browser`  that starts the notebook without opening a browser.

Note that the port `<REMOTE_PORT>` you selected might not be the one that gets assigned to you 
(e.g., in case it’s already being used).

Once `jupyter notebook` has started, it will show you an URL with a security token that you will 
need to open the notebook in your local browser.

On your local terminal:

    ssh -L <LOCAL_PORT>:localhost:<REMOTE_PORT> <REMOTE_USER>@<SERVER_IP>

This command links the `<REMOTE_PORT>` to the specified `<LOCAL_PORT>`.

Once the connection is set up, you can open `jupyter notebook` in your browser by entering: 

    http://localhost:<LOCAL_PORT>/

You may be asked to enter a token (see above).

## TensorBoard

Similar to the jupyter, you can start the `tensorboard` on the server and open it on your local
browser.


On the server run:

    tensorboard --host 0.0.0.0 --logdir <LOG_DIR> --port <REMOTE_PORT>

On the browser of your local computer:

    http://<SERVER_IP>:<REMOTE_PORT>/


## Tmux

In most use-case scenarios, you want:
* to run a script for several hours,
* to run several scripts at the same time.

`tmux` facilitates this functionality.
* To create a new session run `tmux new -s <SESSION_NAME>`.
* To detach from a session press `ctr+b` and then `d`.
* To attach to a session press `tmux a -t <SESSION_NAME>`.
* To easily move across sessions press `ctr+b` and then `s`.

Online resources:
* Do [Tumux tutorial](https://leimao.github.io/blog/Tmux-Tutorial/) to learn more.
* Check the [cheat sheet](https://tmuxcheatsheet.com/) to learn more about the shortcuts.