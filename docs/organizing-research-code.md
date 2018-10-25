## Outline

Organizing the code
- as a python package
   - ‎accessible from anywhere
   - ‎pip install -e . (Makes a big difference)
- ‎snakemake - dag
   - ‎submitting slurm jobs, conda environment
- ‎ipynotebook
  - ‎title=goals
  - ‎section titles
  - enumerate
  - ‎notebooks as a time tracking tool
- ‎R - r notebook, wBuild
  - ‎as an R package?
- ‎writing issues
   - ‎better todo list
   - ‎nicer explainations
   - ‎makes you think
   - ‎can discuss with others
- configuration
   - ‎.env
   - ‎config.py
- ‎data
  - ‎s3 (mention minio)
  - ‎git lfs
  - ‎rsync
  - ‎makefile for syncing
- ‎experiment tracking
   - ‎sacred
   - ‎Config file describing the experiment
- ‎folder structure
  -  ‎most important thing
  -  ‎define it on a piece of paper
  -  ‎stick with it
  -  ‎raw, processed
  -  experiments
  -  ‎always be optimistic
-  ‎editors, terminals etc
   -  ‎byobu + emacs/vim (htop, queue,...)
   -  ‎Zsh
   -  ‎jupyterlab
   -  ‎emacs/other ide's (synthax checking and pep8)


# Scientific working

In this post, I will explain how I organize my research code written in python, give recommendations on different aspects of code organization and list the tools do I use on a day-to-day basis. The folder strcuture was strongly inspired by the `data-science-cookiecutter`. 
- TODO - link to the cookiecutter repository


If you are an R user, I suggest to have a look at the following resources.
 - TODO - refer to wBUild
With R, you have three options with literate programming (code + text + results):
- jupyter notebook with R kernel
- rmarkdown/knitr using the .Rmd or .R
- Rmarkdown notebooks (restricted to Rstudio, jupyter-notebook rmarkdown with inline results like 
- writing an R package [link] - TODO

### Folder structure overview
	
Let's call our research repository: `myproject`. Here is an overview of the file structure:

```
# Python package (shared code)
myproject/
  __init__.py
  __main__.py
  utils.py
  models.py
  datasets.py
  ...
setup.py
conda.env

# Ipython notebooks etc (experiment-specific code)
src/

  exp1/
    1-setup.ipynb
	proto-develop-some-code.ipynb
	2-dataset-exploration.ipynb
	3-modeling.ipynb
	4-evaluation.ipynb
	some_script.py
	some_other_script.R
	Snakefile

	plots/
	  plot1.png
	  plot1.pdf

  exp2/
  exp3/

readme.md
Snakefile
Makefile
.gitignore
LICENSE

# Data (not stored in the repository)
data/  -> softlink to some other location (e.g. not part of the git repository)
  raw/
    exp1/   # used by src/exp1/ code
	  dont-change-me.csv
	exp2/...
	exp3/...
  processed/
    exp1/   # Produced by src/exp1 code
	  reproducible.csv
	exp2/...
	exp3/...
```

## Code

Your code should be stored in a git repository and hosted on a platform supporting issue tracking like Github, Gitlab or Bitbucket.

### `<project>/` Common functions as a python package

`myproject` should contain all the common functions, classes, constants and should be organized as a python package.

```
# Python package (shared code)
myproject/
  # Common functionality among different experiments
  __init__.py  # can be emtpy
  __main__.py
  utils.py
  models.py
  datasets.py
  metrics.py
  plots.py
  preproc.py
  io.py

  # Experiment/dataset specific functionality
  exp/
    __init__.py
    exp1/
	  __init__.py
	  datasets.py
    exp2/
	  __init__.py
	  datasets.py
	  models.py
  ...
setup.py
conda.env
```

#### Setup instructions
1. Structure shared functions/classes into a python package.

2. Install the package using `pip install -e .`. 

3. Put the following three lines to your `~/.ipython/profile_default/ipython_config.py`

```
c.InteractiveShellApp.extensions === ['autoreload']
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
c.InteractiveShellApp.exec_lines.append('print("disable autoreload in ipython_config.py to improve performance.")')
```

This will allow you to import the functions from the python package regardless of where working directory location (`from myprojects.datasets import load_x`). Using the `-e` flag in `pip install` together with the `autoreload` extensions will allow you to edit the `.py` files in the python package while you run the code in the Jupyter notebook and have the changes immediately available (e.g. the python file will be auto-reloaded). Put the required packages from PyPI (e.g. pip-installable) to `setup.py`. Put all the other required pacakges installable using conda to `conda.env` - TODO - link to a conda env. example. 

Try to group the functions/classes based on the functionality they do. Prefer to have less functions/classes per file (even with a single function in the file) rather than too many functions in a single file.

TODO - show a simple `setup.py` and `conda.env` examples.

#### Reading list
- Follow this - TODO tutorial to see how to write a python package.
- Keras repository - very nice code organization

### `src/` - Jupyter notebooks and other scripts

`src/` is a place to put all the analysis code, jupyter notebooks or custom scripts.

```
# Ipython notebooks etc (experiment-specific code)
src/

  exp1/
    1-setup.ipynb
	proto-develop-some-code.ipynb
	2-dataset-exploration.ipynb
	3-modeling.ipynb
	4-evaluation.ipynb
	some_script.py
	train.py
	evaluate.py
	some_other_script.R
	Snakefile

	plots/
	  plot1.png
	  plot1.pdf

  exp2/
  exp3/
```

Research projects often consist of a few sub-projects. These are typically based on different datasets. In the example above, the sub-projects are `exp1`, `exp2` and `exp3`.

For each sub-project I end up writing a bunch of jupyter-notebooks together with scripts and ideally a Snakefile.

#### Tips for writing jupyter notebooks

- enumerate them (e.g. increment the number in the filename for every notebook you create - `02-my-second-notebook.ipynb`)
  - that way the notebooks will serve as a lab-book, giving you a sense of how the notebooks evolved through time.
- first cell of the notebook should list the goals of the notebook in a sentence or two
  - e.g. train a model for X, investigate the relationship between X and Y
- second cell should contain a TODO section clearly listing the steps that need to be done in order to achieve the goal
- Section title should be the items from the TODO list. That way, it is clear what you are trying achieve in every sub-section of the notebook. (I learn this TODO->section titles trick from Jeremy Howard's fast.ai course)
- After you are done running the code, write a 'Conclusions' cell after the 'Goals' sections
  - That way, when you re-visit the notebook you can clearly see what was the original goal of the notebook and what the outcome/onclusions
- As your codebase in the jupyter notebook expands, package it into functions/classes and put them to the python package.
  - This will make the notebook more readable.
  - You can re-use the functions in other notebooks.

Example structure of the jupyter notebook:
```
## Goals
- train a model for X

## Conclusions
- Final model performance was:
  - auPRC: 0.45
- X could be improved
- Y doesn't work

## TODO
- [x] get the data
- [x] train a simple model
- [x] evaluate the model
- [x] train a more complex model

## Get the data
...

## Train a simple model
...

## Evaluate the model
...

## Train a more complex model
...
```

### `Snakefile`

If you are not familiar with [Snakemake](TODO), go ahead and learn how to use it by following Getting started TODO. Try to use it as much as possible. It will make your life much easier down the road when you have to say re-run the same scripts on new or updated data. As soon as you have to run one script/command multiple times, consider using snakemake. Learn how to use it with the job scheduler of your local cluster (e.g. SLURM). Also, consider using separate conda environments for different rules.

Snakemake will enforce many good practices including a clear directory structure.

## `data/`

```
data/  -> softlink to some location with more storage available like /mnt/data/basepair
  raw/
    exp1/   # used by src/exp1/ code
	  dont-change-me.csv
	exp2/...
	exp3/...
  processed/
    exp1/   # Produced by src/exp1 code
	  reproducible.csv
	exp2/...
	exp3/...
```

### Separate data from code (e.g. don't put data to other folder than `data/`)

It can be convenient to start writing the output of the scripts in the same folder as the scripts itself. Don't be lazy, always write the results to the dedicated data folder `data/processed/...`. That way, you will clearly separate code from data.

### Strictly separate raw from processed data

Files in `raw/` should contain any files which you obtained externally (e.g. you downloaded it from the web, a collaborator emailed it to you, etc). Raw data should stay, as the name suggests, raw. You should never manipulate it.

Files in `processed/` should should contain everything else not falling into the `raw/` category. These will be things like pre-processed training data, trained model files, evaluation files, plots etc. Ideally, you should be able to reproduce all the files here given the code/commands and the `raw/` files.

### Store the data outside of your home folder and softlink it to `data/`

While the code repository should be stored in your home folder like `~/workspace/basepair/`, the data itself can or should be stored in a different folder. Often, your home folder will be backed-up nightly on a server hence there will typically be a size restriction on your home folder.

To have a consistent path regardless of the actuall storage directory, use a softlink: `~/workspace/basepair/data -> /mnt/data/basepair`. All your scripts should refer to the `data` softlink rather than the original `/mnt/data/basepair`. That way, you can change the location of your folder later (in case the disk gets full for example) without having to change your code.


### Be optimistic about your folder structure. Use rather too many folder levels than too few.

The only structure I've imposed so far on the `data/` folder is to separate raw from processed data. From there on, you should think carefully beforehand how to best organize your folder structure.
Since research is a messy process where you end up trying many different things, your folder structure should be *ambitious* accordingly.

E.e. Instead of dumping all the files into one folder

```
exp1-trained-model1.hdf4
exp1-model1-results.json
exp1-model1-plot1.pdf
exp1-predictions_train-model1.csv
exp1-predictions_valid-model1.csv
exp1-trained-model2.hdf4
exp1-model2-results.json
exp1-model2-plot1.pdf
exp2-trained-model1.hdf4
exp2-model1-results.json
exp2-model1-plot1.pdf
...
```

design the folder structure beforehand. I suggest doing this on a piece of paper. Being ambitious meens that you design the folder structure for the case of trying many different things. Each folder level should represent one logical unit. Once you are done designinig the folder structure, you should write the design to the script header or a readme file. I like to write it down in the following style:

```
Output folder structure:
------------------------
<exp>/<id>/hparams.json
<exp>/<id>/dataspec.json
<exp>/<id>/model.hdf5
<exp>/<id>/training_history.csv
<exp>/<id>/eval/<split>/metrics/model1-results.json
<exp>/<id>/eval/<split>/plots/plot1.pdf
<exp>/<id>/eval/<split>/plots/plot2.pdf
<exp>/<id>/eval/<split>/predictions.hdf5

exp: experiment name
id: different model instances
split: data split where to evaluate the model on
```

You will realize that when you start using Snakemake, you will be forced to have a clear directory structure.

### Where to store the data and how to share it?

While it's obvious that code should be shared using some git platorm like GitHub, Gitlab or Bitbucket, it's not obvious how to share the data (with a collegue or yourself on a different machine).

Depending with whom do you collaborate and how large your data are, you might use different solutions for storing data.

- shared file system (ala NFS)
  - who: collegue from the same research group
  - where: one of your servers
- [rsync](https://linux.die.net/man/1/rsync)
  - who:
    - you
	  - working from home
	  - from a different machine without a shared file system
	  - having to move files to a fast local ssd drive
  - where: one of your servers
  - tip: consider setting up a .rsync_ignore file where you can exclude large raw files
  - Typical flag for rsync I use: `rsync -av --progress`
    - It's worth reading the rsync man page. It reads like a book.
  - Prefer using `rsync` over `cp -R` or `scp -R`
- sshfs mount
  - who: you, working from home
  - where: one of your servers
  - can be slow
  - jobs are not io intensive
- S3-like object storage (Amazon S3, Google Cloud storage (GCS), Minio)
  - who: external collaborator
    - Link to the tutorial for installing the aws command
  - Your collaborator has to have an AWS, gCloud account.
  - A nice self-hosted S3 option is <https://www.minio.io/>.
  - where: Cloud
- Git-LFS
  - who: external collaborator
  - where: Github
  - relatively small (~GB) data
  - will be treated in the same way as the code (e.g. you can `git pull` the data, it will be versioned, ...)

  
Regardless of which storage you use, I highly recommend putting the common commands to pull/push or mount/unmount the data to a Makefile. 
  - TODO - link to the Makefile with all the commands

- TODO - refer to the cookiecutter 

## Configuration

For non-sensitive configurations used from python, put all the constants to `<project>/config.py`. I like to put a `get_data_dir` function to `config.py`:

```python
def get_data_dir():
    """Returns the data directory
    """
    import inspect
    import os
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_path = os.path.dirname(os.path.abspath(filename))
    DATA = os.path.join(this_path, "../data")
    if not os.path.exists(DATA):
        raise ValueError(DATA + " folder doesn't exist")
    return DATA
```

Then on top of every jupyter notebook, I will have the following code:

```python
from basepair.config import get_data_dir
ddir = get_data_dir()
```

This allows me to specify the path to the data directory regardless of the working directory.

In case the configuration is sensitive (like private keys), use [dotenv](https://github.com/theskumar/python-dotenv). E.e. use environment variables (defined say in .bashrc) or 
use the `.env` files. You can load those variables within the `config.py` module.

## Keeping track of the TODO's, ideas and results using git issues

### Git issues

Use git-issues to keep track of the TODO's and also experimental results. They allows you to explain your ideas or visualize your results in great detail.

I see four main advantages of using git issues to track progress of the project:
1. writing git issues before you start working makes you think of what exactly are you going to do and why you are doing it. 
2. you can discuss the ideas and results with your collaborators.
3. You can save your results as comments and keep all the information and discussions in one place
  - even when you close the issue, you can re-visit this information
4. You can easily add screenshots (e.g. plots/tables) as comments
  - Save a screenshot to clipboard and then directly paste the figure into the issue comment.

### Milestones

For larger projects, it can be useful to create a [milestone](https://help.github.com/articles/about-milestones/) (say paper submission) and select all the issues that need to be closed for that milestone.

## Other tips

- Setup ssh keys and `~/.ssh/config`
  - "I have no time" is not a valid excuse for lacking the passwordless ssh access. You will save time in the long run.
- Use Python 3 (Python 2 will not be supported after 2020)
- Use [Jupyter lab](https://github.com/jupyterlab/jupyterlab) instead of [Jupyter notebook](https://github.com/jupyterlab/jupyterlab)
  - It's amazing!
- Commit frequently (multiple times a day)
- Use Byobu or tmux instead of screen
  - Learn how to split windows and use tabs
  - Have a htop, `watch nvidia-smi` session always running in some tab
- Use zsh together with [oh-my-zsh](https://ohmyz.sh/) instead of bash. 
  - Even the auto-completion feature (cycle through options using tab without re-printing all the options) alone is enough to make the switch IMO.
- Learn how to use the emacs/vim properly for work from the terminal.
- Use pep8 style checker in your text editor (emacs/pycharm/atom/vim/sublime/...) when editing the python package files.
- Editing code on the remote machine
  - through terminal via emacs/vim
    - use for quick edits like script error fixing
  - through jupyter lab
    - I typically end up copying code from the notebooks to the python files
  - sshfs mount to edit code on the remote machine
    - This allows you to fully leverage the local text editor (emacs+elpy in my case)


### Dependency installation - use (mini)conda

Install miniconda/anaconda and create a separate conda environment `<project name>` for the project. This will clearly separate the environments (and thereby package versions) between different projects.
