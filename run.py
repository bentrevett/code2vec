import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from tqdm import tqdm_notebook as tqdm
import os
import random
import pickle

# setup parameters

SEED = 1234
DATA_DIR = 'data'
DATASET = 'java14m'
EMBEDDING_DIM = 128
DROPOUT = 0.25
BATCH_SIZE = 1024
CHUNKS = 10
MAX_LENGTH = 200
LOG_EVERY = 1000 #print log of results after every LOG_EVERY batches
N_EPOCHS = 20
LOG_DIR = 'logs'
SAVE_DIR = 'models'
LOG_PATH = os.path.join(LOG_DIR, f'{DATASET}-log.txt')
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{DATASET}-model.pt')

# set random seeds

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# load counts of each token in dataset

with open(f'{DATA_DIR}/{DATASET}/{DATASET}.dict.c2v', 'rb') as file:
    word2count = pickle.load(file)
    path2count = pickle.load(file)
    target2count = pickle.load(file)
    n_training_examples = pickle.load(file)

# create vocabularies

word2idx = {'<unk>': 0, '<pad>': 1}
path2idx = {'<unk>': 0, '<pad>': 1 }
target2idx = {'<unk>': 0, '<pad>': 1}

idx2word = {}
idx2path = {}
idx2target = {}

for w in word2count.keys():
    word2idx[w] = len(word2idx)
    
for k, v in word2idx.items():
    idx2word[v] = k
    
for p in path2count.keys():
    path2idx[p] = len(path2idx)
    
for k, v in path2idx.items():
    idx2path[v] = k
    
for t in target2count.keys():
    target2idx[t] = len(target2idx)
    
for k, v in target2idx.items():
    idx2target[v] = k
    
class Attention(nn.Module):
    def __init__(self, nodes_dim, paths_dim, embedding_dim, output_dim, dropout):
        super().__init__()
        
        self.node_embedding = nn.Embedding(nodes_dim, embedding_dim)
        self.path_embedding = nn.Embedding(paths_dim, embedding_dim)
        self.W = nn.Parameter(torch.randn(1, embedding_dim, 3*embedding_dim))
        self.a = nn.Parameter(torch.randn(1, embedding_dim, 1))
        self.out = nn.Linear(embedding_dim, output_dim)
        self.do = nn.Dropout(dropout)
        
    def forward(self, starts, paths, ends):
        
        #all should be [bsz, length]
        
        W = self.W.repeat(starts.shape[0], 1, 1)
        
        # bsz, d, d*3
        
        embedded_starts = self.node_embedding(starts)
        embedded_paths = self.path_embedding(paths)
        embedded_ends = self.node_embedding(ends)
        
        c = torch.cat((embedded_starts, embedded_paths, embedded_ends), dim=2)
        
        # bsz, len, d*3
        
        c = self.do(c)
        
        # bsz, len, d*3
        
        c = c.permute(0, 2, 1)
        
        # bsz, d*3, len

        x = torch.tanh(torch.bmm(W, c))
        
        # bsz, d, len
        
        x = x.permute(0, 2, 1)
        
        #bsz, len, d
        
        a = self.a.repeat(starts.shape[0], 1, 1)
        
        z = torch.bmm(x, a).squeeze(2)
        
        #bsz, len
        
        z = F.softmax(z, dim=1)
        
        #bsz, len
        
        z = z.unsqueeze(2)
        
        #bsz, len, 1
        
        x = x.permute(0, 2, 1)
        
        #bsz, d, len
        
        v = torch.bmm(x, z).squeeze(2)
        
        #bsz, d
        
        out = self.out(v)
        
        return out

model = Attention(len(word2idx), len(path2idx), EMBEDDING_DIM, len(target2idx), DROPOUT)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda')

model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(fx, y):
    pred_idxs = fx.max(1, keepdim=True)[1]
    correct = pred_idxs.eq(y.view_as(pred_idxs)).sum()
    acc = correct.float()/pred_idxs.shape[0]
    return acc

def calculate_f1(fx, y):
    pred_idxs = fx.max(1, keepdim=True)[1]
    pred_names = [idx2target[i.item()] for i in pred_idxs]
    original_names = [idx2target[i.item()] for i in y]
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(pred_names, original_names):
        predicted_subtokens = p.split('|')
        original_subtokens = o.split('|')
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1

def parse_line(line):
    name, *tree = line.split(' ')
    tree = [t.split(',') for t in tree if t != '' and t != '\n']
    return name, tree

def train(model, file_path, optimizer, criterion):
    
    n_batches = 0
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0
    
    model.train()
    
    examples = []
    
    with open(file_path, 'r') as f:
        
        for line in f:
            #each line is an example

            #each example is made of the function name and then a sequence of triplets
            #the triplets are (node, path, node)

            example_name, example_body = parse_line(line)
            
            #max length set while preprocessing

            assert len(example_body) <= MAX_LENGTH, print(len(example_body))
            
            #need to pad all to maximum length

            example_length = len(example_body)
            
            example_body += [['<pad>', '<pad>', '<pad>']]*(MAX_LENGTH - len(example_body))
            
            assert len(example_body) == MAX_LENGTH, print(example_body)
            
            #add example to all examples

            examples.append((example_name, example_body, example_length))
            
            #check if we have enough examples to do some training
            
            if len(examples) >= (BATCH_SIZE * CHUNKS):
                
                #if we do, want them in a random order
                
                random.shuffle(examples)
                
                #for each random chunk of BATCH_SIZE examples
                
                for i in range(CHUNKS):
                    
                    #get the raw data
                    
                    raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
                    
                    #create a tensor to store the batch
                    
                    tensor_n = torch.zeros(BATCH_SIZE).long() #name
                    tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #left node
                    tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #path
                    tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #right node
                    mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float() #mask
                    
                    #for each example in our raw data
                    
                    for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
                        
                        #convert to idxs using vocab
                        #use <unk> tokens if item doesn't exist inside vocab
                        
                        temp_n = target2idx.get(name, target2idx['<unk>'])
                        temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
                        
                        #store idxs inside tensors

                        tensor_n[j] = temp_n
                        tensor_l[j,:] = torch.LongTensor(temp_l)
                        tensor_p[j,:] = torch.LongTensor(temp_p)
                        tensor_r[j,:] = torch.LongTensor(temp_r)   
                     
                        #create masks

                        mask[j, length:] = 0
                        
                    #place on gpu

                    tensor_n = tensor_n.to(device)
                    tensor_l = tensor_l.to(device)
                    tensor_p = tensor_p.to(device)
                    tensor_r = tensor_r.to(device)
                        
                    #put into model
                    
                    optimizer.zero_grad()
                    
                    fx = model(tensor_l, tensor_p, tensor_r)
                    
                    del tensor_l
                    del tensor_p
                    del tensor_r
                    
                    loss = criterion(fx, tensor_n)
                    
                    acc = calculate_accuracy(fx, tensor_n)
                    precision, recall, f1 = calculate_f1(fx, tensor_n)
                    
                    del tensor_n
                    
                    loss.backward()
        
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                    epoch_p += precision
                    epoch_r += recall
                    epoch_f1 += f1
                    
                    del loss
                    del acc
                    
                    n_batches += 1
                                        
                    if n_batches % LOG_EVERY == 0:
                
                        loss = epoch_loss / n_batches
                        acc = epoch_acc / n_batches
                        precision = epoch_p / n_batches
                        recall = epoch_r / n_batches
                        f1 = epoch_f1 / n_batches

                        log = f'\t| Batches: {n_batches} | Completion: {((n_batches*BATCH_SIZE)/n_training_examples)*100:03.3f}% |\n'
                        log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                        with open(LOG_PATH, 'a+') as f:
                            f.write(log+'\n')
                        print(log)

                examples = []
                                
            else:
                pass
      
    #outside of for line in f, but will still have some examples left over

    random.shuffle(examples)
            
    #how many batches can we make?

    n = len(examples)//BATCH_SIZE
    
    for i in range(n):
        
        #get the raw data
        
        raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
        
        #create a tensor to store the batch
        
        tensor_n = torch.zeros(BATCH_SIZE).long() #name
        tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #left node
        tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #path
        tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #right node
        mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float() #mask
        
        #for each example in our raw data
        
        for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
            
            #convert to idxs using vocab
            #use <unk> tokens if item doesn't exist inside vocab
            
            temp_n = target2idx.get(name, target2idx['<unk>'])
            temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
            
            #store idxs inside tensors

            tensor_n[j] = temp_n
            tensor_l[j,:] = torch.LongTensor(temp_l)
            tensor_p[j,:] = torch.LongTensor(temp_p)
            tensor_r[j,:] = torch.LongTensor(temp_r)   
            
            #create masks

            mask[j, length:] = 0
            
        #place on gpu

        tensor_n = tensor_n.to(device)
        tensor_l = tensor_l.to(device)
        tensor_p = tensor_p.to(device)
        tensor_r = tensor_r.to(device)
            
        #put into model
        
        optimizer.zero_grad()
        
        fx = model(tensor_l, tensor_p, tensor_r)
        
        del tensor_l
        del tensor_p
        del tensor_r
        
        loss = criterion(fx, tensor_n)
        
        acc = calculate_accuracy(fx, tensor_n)
        precision, recall, f1 = calculate_f1(fx, tensor_n)
        
        del tensor_n
        
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_p += precision
        epoch_r += recall
        epoch_f1 += f1
        
        del loss
        del acc
        
        n_batches += 1

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

def evaluate(model, file_path, criterion):
    
    n_batches = 0
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_r = 0
    epoch_p = 0
    epoch_f1 = 0
    
    model.eval()
    
    examples = []
    
    with torch.no_grad():
    
        with open(file_path, 'r') as f:

            for line in f:

                example_name, example_body = parse_line(line)

                assert len(example_body) <= MAX_LENGTH, print(len(example_body))

                example_length = len(example_body)

                example_body += [['<pad>', '<pad>', '<pad>']]*(MAX_LENGTH - len(example_body))

                assert len(example_body) == MAX_LENGTH, print(example_body)

                examples.append((example_name, example_body, example_length))

                #check if we have enough examples to do some training

                if len(examples) >= (BATCH_SIZE * CHUNKS):

                    #if we do, want them in a random order

                    random.shuffle(examples)

                    #for each random chunk of BATCH_SIZE examples

                    for i in range(CHUNKS):

                        #get the raw data

                        raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE*i:BATCH_SIZE*(i+1)])

                        #create a tensor to store the batch

                        tensor_n = torch.zeros(BATCH_SIZE).long()
                        tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()
                        tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()
                        tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long()
                        mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float()

                        #for each example in our raw data

                        for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):

                            #convert to idxs

                            temp_n = target2idx.get(name, target2idx['<unk>'])
                            temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])

                            tensor_n[j] = temp_n

                            tensor_l[j,:] = torch.LongTensor(temp_l)
                            tensor_p[j,:] = torch.LongTensor(temp_p)
                            tensor_r[j,:] = torch.LongTensor(temp_r)   

                            mask[j, length:] = 0

                        #place on gpu
                        tensor_n = tensor_n.to(device)
                        tensor_l = tensor_l.to(device)
                        tensor_p = tensor_p.to(device)
                        tensor_r = tensor_r.to(device)

                        #put into model
                        
                        fx = model(tensor_l, tensor_p, tensor_r)

                        del tensor_l
                        del tensor_p
                        del tensor_r

                        loss = criterion(fx, tensor_n)

                        acc = calculate_accuracy(fx, tensor_n)
                        precision, recall, f1 = calculate_f1(fx, tensor_n)

                        del tensor_n

                        epoch_loss += loss.item()
                        epoch_acc += acc.item()
                        epoch_p += precision
                        epoch_r += recall
                        epoch_f1 += f1

                        del loss
                        del acc

                        n_batches += 1

                        if n_batches % LOG_EVERY == 0:

                            loss = epoch_loss / n_batches
                            acc = epoch_acc / n_batches
                            precision = epoch_p / n_batches
                            recall = epoch_r / n_batches
                            f1 = epoch_f1 / n_batches

                            log = f'\t| Batches: {n_batches} |\n'
                            log += f'\t| Loss: {loss:02.3f} | Acc.: {acc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}'
                            with open(LOG_PATH, 'a+') as f:
                                f.write(log+'\n')
                            print(log)

                    examples = []

                else:
                    pass
      
        #outside of for line in f, but will still have some examples left over

        random.shuffle(examples)
                
        #how many batches can we make?

        n = len(examples)//BATCH_SIZE
        
        for i in range(n):
            
            #get the raw data
            
            raw_batch_name, raw_batch_body, batch_lengths = zip(*examples[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            
            #create a tensor to store the batch
            
            tensor_n = torch.zeros(BATCH_SIZE).long() #name
            tensor_l = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #left node
            tensor_p = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #path
            tensor_r = torch.zeros((BATCH_SIZE, MAX_LENGTH)).long() #right node
            mask = torch.ones((BATCH_SIZE, MAX_LENGTH)).float() #mask
            
            #for each example in our raw data
            
            for j, (name, body, length) in enumerate(zip(raw_batch_name, raw_batch_body, batch_lengths)):
                
                #convert to idxs using vocab
                #use <unk> tokens if item doesn't exist inside vocab
                
                temp_n = target2idx.get(name, target2idx['<unk>'])
                temp_l, temp_p, temp_r = zip(*[(word2idx.get(l, word2idx['<unk>']), path2idx.get(p, path2idx['<unk>']), word2idx.get(r, word2idx['<unk>'])) for l, p, r in body])
                
                #store idxs inside tensors

                tensor_n[j] = temp_n
                tensor_l[j,:] = torch.LongTensor(temp_l)
                tensor_p[j,:] = torch.LongTensor(temp_p)
                tensor_r[j,:] = torch.LongTensor(temp_r)   
                
                #create masks

                mask[j, length:] = 0
                
            #place on gpu

            tensor_n = tensor_n.to(device)
            tensor_l = tensor_l.to(device)
            tensor_p = tensor_p.to(device)
            tensor_r = tensor_r.to(device)
                
            #put into model
            
            fx = model(tensor_l, tensor_p, tensor_r)
            
            del tensor_l
            del tensor_p
            del tensor_r
            
            loss = criterion(fx, tensor_n)
            
            acc = calculate_accuracy(fx, tensor_n)
            precision, recall, f1 = calculate_f1(fx, tensor_n)
            
            del tensor_n

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_p += precision
            epoch_r += recall
            epoch_f1 += f1
            
            del loss
            del acc
            
            n_batches += 1

    return epoch_loss / n_batches, epoch_acc / n_batches, epoch_p / n_batches, epoch_r / n_batches, epoch_f1 / n_batches

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

if not os.path.isdir(f'{LOG_DIR}'):
    os.makedirs(f'{LOG_DIR}')

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

for epoch in range(N_EPOCHS):

    log = f'Epoch: {epoch+1:02} - Training'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)
    
    train_loss, train_acc, train_p, train_r, train_f1 = train(model, f'{DATA_DIR}/{DATASET}/{DATASET}.train.c2v', optimizer, criterion)
    
    log = f'Epoch: {epoch+1:02} - Validation'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)
    
    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, f'{DATA_DIR}/{DATASET}/{DATASET}.val.c2v', criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    log = f'| Epoch: {epoch+1:02} |\n'
    log += f'| Train Loss: {train_loss:.3f} | Train Precision: {train_p:.3f} | Train Recall: {train_r:.3f} | Train F1: {train_f1:.3f} | Train Acc: {train_acc*100:.2f}% |\n'
    log += f'| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p:.3f} | Val. Recall: {valid_r:.3f} | Val. F1: {valid_f1:.3f} | Val. Acc: {valid_acc*100:.2f}% |'
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

log = 'Testing'
with open(LOG_PATH, 'a+') as f:
    f.write(log+'\n')
print(log)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc, test_p, test_r, test_f1 = evaluate(model, f'{DATASET}/{DATASET}/{DATASET}.test.c2v', criterion)

log = f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p:.3f} | Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f} | Test Acc: {test_acc*100:.2f}% |'
with open(LOG_PATH, 'a+') as f:
    f.write(log+'\n')
print(log)
