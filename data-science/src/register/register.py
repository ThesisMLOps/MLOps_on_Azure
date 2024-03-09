# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import pickle
import mlflow

import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main(args):
    '''Loads model, registers it if deply flag is True'''

    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
        
    mlflow.log_metric("deploy flag", int(deploy_flag))
    deploy_flag=1
    if deploy_flag==1:

        print("Registering ", args.model_name)

        # load model
        model = mlflow.sklearn.load_model(args.model_path)

        conda_env = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.8",
                {
                    "pip": [
                        "scikit-learn==1.2.2",
                        'cloudpickle==2.2.1',
                        'numpy==1.24.4',
                        'packaging==23.2',
                        'psutil==5.9.6',
                        'pyyaml==6.0.1',
                        'scipy==1.10.1',
                        'mlflow'                                               
                        ],
                    },
                ],
            }

        #conda_env = {
        #    'channels': ['conda-forge'],
        #    'dependencies': [
        #       'python=3.8.18',
        #        'pip'],
        #        'pip': [
        #            'mlflow',
        #            'scikit-learn==1.2.2',
        #            'cloudpickle==2.2.1',
        #            'numpy==1.24.4',
        #            'packaging==23.2',
        #            'psutil==5.9.6',
        #            'pyyaml==6.0.1',
        #            'scipy==1.10.1'
        #            ],
        #            'name': 'mlflow-env'
        #            } 

        # log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name, conda_env=conda_env)

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Model will not be registered!")

if __name__ == "__main__":

    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
