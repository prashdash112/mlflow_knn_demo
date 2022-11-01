import mlflow
import os
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    
    '''
    
    Function to compile, bind and run the mlflow componenet
    
    '''

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "comp1"),
        "main",
        parameters={
            "neighbour": config["KNN"]["neighbour"],
            "weights": config["KNN"]["weights"],
            "leaf": config["KNN"]["leaf"]
        },
    )

if __name__ == "__main__":
    go()




