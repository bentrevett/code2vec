import os
import shutil
import subprocess

data_dir = 'data'
datasets = ['java-small', 'java-med']

MAX_SIZE = 60

for dataset in datasets:

    for t in ['validation', 'test', 'training']:

        projects = os.listdir(f'{data_dir}/{dataset}/{t}')

        for p in projects:

            project_path = f'{data_dir}/{dataset}/{t}/{p}'

            output = subprocess.check_output(f'du -sh {project_path}', shell=True).decode('utf-8').split('\t')[0]
        
            if output[-1] == 'K':
                continue

            if output[-1] == 'M':
                size = float(output[:-1])
            
                if size > MAX_SIZE:
                    n_directories = int(size / MAX_SIZE) 
                    print(f'Balancing {project_path} which is currently {output} into {n_directories} extra directories')  
                    for i in range(n_directories):
                        if os.path.exists(f'{project_path}{i}'):
                            while os.path.exists(f'{project_path}{i}'):
                                i += 1
                            os.mkdir(f'{project_path}{i}')
                        else:
                            os.mkdir(f'{project_path}{i}')
                        files = os.listdir(f'{project_path}')
                        files_to_move = files[:int(len(files)/(n_directories+1))]
                        for f in files_to_move:
                            shutil.move(f'{project_path}/{f}', f'{project_path}{i}/{f}')

            else:
                assert 1 == 2
                print(f'{project_path} = {output}')

        
