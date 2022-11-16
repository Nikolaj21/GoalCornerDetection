import os
import sys

## ALL relevant and often used paths will be put here
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'Data')


def export_wandb_api():
    os.environ['WAND_API_KEY'] = "fd61955568b424c577dffeebaec0f1a50f1d73be"
    return print("# Wandb api key successfully exported to environment #")

#FIXME won't work in practice, because we won't be able to use this without import utils, which won't work across platforms unless this function has already been run.... sad....
def export_project_path():
    """
    Appends the project path to sys, so files can be easily imported.

    BEWARE! For this to work across systems, it requires that this file (i.e. utils.py)
    is located directly in the Project folder (i.e in the folder GoalCornerDetection)
    """
    projectpath = os.getcwd()
    sys.path.append(projectpath)