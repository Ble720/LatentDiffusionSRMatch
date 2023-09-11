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
    return fv[None, :]

def feature_matrix(flop, pool, save):
    fm = None

    if type(flop) == str:
        for p in os.listdir(flop):
            full_path = flop + '/' + p
            lat_matrix = torch.load(full_path.replace('/', os.sep))
            v = vectorize(lat_matrix, pool)
            if fm == None:
                fm = v
            else:
                fm = torch.cat([fm, v], dim=0)
    else:
        for f in flop:
            v = vectorize(f, pool)
            if fm == None:
                fm = v
            else:
                fm = torch.cat([fm, v], dim=0)

    if save:
        torch.save(fm, save + '/features.pt')

    print(f'Combined {fm.shape[0]} .pt files to a {fm.shape} tensor\n')
    return fm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--pool', type=str, default='none')
    opt = parser.parse_args()

    if not opt.source or not opt.save:
        raise Exception("No source folder or save path provided")
    
    #create save directory if it doesn't exist
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
        
    feature_matrix(opt.source, opt.pool, opt.save)