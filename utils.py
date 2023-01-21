import os

## ALL relevant and often used paths will be put here
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'Data')
# DATA_DIR_TEST = '../Scratch/s163848/Data_test'
DATA_DIR_TEST = '/zhome/60/1/118435/Master_Thesis/Scratch/s163848/Data_test'

def export_wandb_api():
    os.environ['WANDB_API_KEY'] = "fd61955568b424c577dffeebaec0f1a50f1d73be"
    return print("# Wandb api key successfully exported to environment #")
