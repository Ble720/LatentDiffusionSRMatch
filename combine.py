import torch
import os
import argparse

def vectorize(latent_in, pool):
    if pool == 'max':
        pool = torch.nn.MaxPool2d(4, padding=2, stride=2)
        output = pool(latent_in)
    elif pool == 'avg':
        pool = torch.nn.AvgPool2d(4, padding=2, stride=2)
        output = pool(latent_in)
    else:
        output = latent_in
    fv = torch.flatten(output).detach().clone()
    return fv

def feature_matrix(path, pool, save):
    fv = []
    for i in range(len(path)):
        full_path = path + '/' + str(i) + '.pt'
        lat_matrix = torch.load(full_path)
        v = vectorize(lat_matrix, pool)
        fv.append(v)
    
    fm = torch.cat(fv, dim=0)

    if save:
        torch.save(fm, save + '/features.pt')

    print(f'Combined {len(fv)} .pt files to a {fm.shape} tensor\n')
    return fm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--pool', type=str, defaul='none')
    opt = parser.parse_args()

    if not opt.source or not opt.save:
        raise Exception("No source folder or save path provided")
    
    #create save directory if it doesn't exist
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
        
    feature_matrix(opt.source, opt.pool, opt.save)