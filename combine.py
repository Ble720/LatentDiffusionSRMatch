import torch
import os
import argparse

def combine_features(paths, sv_path):
    all_path = os.listdir(paths)
    all_features = []
    for i in range(len(all_path)):
        full_path = paths + '/' + str(i) + '.pt'
        feat = torch.load(full_path)
        if len(feat.shape) == 1:
            feat = torch.unsqueeze(feat, dim=0)
        all_features.append(feat)
    features = torch.cat(all_features, dim=0)
    torch.save(features, sv_path + '/features.pt')
    print(f'Combined {len(all_features)} .pt files to a {features.shape} tensor\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--save', type=str)
    opt = parser.parse_args()
    if not opt.source or not opt.save:
        raise Exception("No source folder or save path provided")
    
    #create save directory if it doesn't exist
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
        
    combine_features(opt.source, opt.save)