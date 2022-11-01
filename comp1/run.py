import os
import logging
import numpy as np 
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    '''
    
    Implementation of K- Nearest Neighbour Algorithm using best MLOPS practices

    '''
    logger.info('Innitiating WandB Run......')

    run = wandb.init(project='KNN_1',job_type='dev',save_code=True)

    logger.info('Importing the dataframe')

    df = pd.read_csv(r'./datasets/KNN_Project_Data')
    
    logger.info('Performing and logging EDA....')
    
    pair_plot = sns.pairplot(df, hue='TARGET CLASS' ,kind='scatter')
    pair_plot.figure.savefig(r'./eda/output.png')

    logger.info('Standard scaling the dataset to make std dev=1 and mean=0')
    
    scaler = StandardScaler()
    scaler.fit(df.drop('TARGET CLASS', axis=1))
    scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))   

    logger.info('Creating the dependent features df.....')
    
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

    logger.info('Performing Train-Test split...')
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'],test_size=0.3)

    logger.info('Building KNN model to identify target_class...')
    
    neighbour = KNeighborsClassifier(n_neighbors=args.neighbour, weights=args.weights, leaf_size=args.leaf)
    
    logger.info('Fitting the data to model..')
    
    fit = neighbour.fit(X_train,y_train)

    logger.info('Predicting the test results..')
    
    predict = neighbour.predict(X_test)

    logger.info('Generating Classification Report...')
    
    report = classification_report(y_test, predict, output_dict=True)
    

    wandb.log(
        {
            "Pair plot": wandb.Image(r'./eda/output.png')
        }
    )
    predictions = pd.DataFrame(predict, columns=['Prediction'])
    predictions.to_csv(r'./datasets/predictions.csv')
    artifact = wandb.Artifact('Predictions', type='dataset')
    artifact.add_file(r'./datasets/predictions.csv')
    wandb.log_artifact(artifact)

    return True

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Training a KNN model and tracking it with WandB',
        fromfile_prefix_chars="@")
    
    parser.add_argument(
        '--neighbour',
        type=int,
        help='The number of neighbour values',
        default=5
        )
    
    parser.add_argument(
        '--weights',
        type=str,
        help='Weight of KNN algorithm',
        default='uniform'
        )
    
    parser.add_argument(
        '--leaf',
        type=int,
        help='The leaf size in KNN algorithm',
        default=30
        )
    
    args = parser.parse_args()
    go(args)

