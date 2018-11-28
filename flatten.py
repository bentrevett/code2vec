import os
import subprocess

data_dir = 'data'
datasets = ['java-small', 'java-med']

for dataset in datasets:
    
    for t in ['validation', 'test', 'training']:

        projects = os.listdir(f'{data_dir}/{dataset}/{t}')

        for p in projects:

            print(f'Flattening {data_dir}/{dataset}/{t}/{p}')

            subprocess.run(f"find {data_dir}/{dataset}/{t}/{p}/ -mindepth 2 -type f -exec mv '{{}}' {data_dir}/{dataset}/{t}/{p} \;", shell=True)
