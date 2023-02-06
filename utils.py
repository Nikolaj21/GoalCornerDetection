import os

## ALL relevant and often used paths will be put here
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'Data')
DATA_DIR_TEST = '/zhome/60/1/118435/Master_Thesis/Scratch/s163848/Data_test'

def export_wandb_api():
    os.environ['WANDB_API_KEY'] = open('wandb_api.txt','r').read()
    print(os.environ['WANDB_API_KEY'])
    return print("# Wandb api key successfully exported to environment #")
