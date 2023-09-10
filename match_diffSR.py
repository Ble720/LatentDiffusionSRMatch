import requests

from PIL import Image
from io import BytesIO
from SR_pipe import LDMSuperResolutionPipeline
from features import get_features

import torch
import torch.nn.functional as F
import os
import shutil
import numpy as np
import argparse

def get_img_paths(folder_path):
  names = []
  for fRoot, fDirs, fFiles in os.walk(folder_path):
    for ffile in fFiles:
      full_path = os.path.join(fRoot, ffile).replace('\\', '/')
      names.append(full_path)
  return names   

def similarity(source_fm, target_fm, mode):
  if mode == 'L2':
    return torch.cdist(source_fm, target_fm)
  elif mode == 'cos':
    return cos_sim(source_fm, target_fm)
  elif mode == 'dot':
    return dot_sim(source_fm, target_fm)

def cos_sim(source_fm, target_fm):
    # cosine similarity of feature vectors
    src = F.normalize(source_fm, dim=1)
    trg = F.normalize(target_fm, dim=1)
    score = torch.matmul(src, trg.T)
    return score

def dot_sim(source_fm, target_fm):
  score = torch.matmul(source_fm, target_fm.T)
  return score

def run_comparison(model, source_feat, source_path, target_feat, target_path, sv_path, mode, topk, num_step=100):
  source_name = get_img_paths(source_path)
  target_name = get_img_paths(target_path)

  if source_feat.endswith('.pt'):
    source_features = torch.load(source_feat) 
  else:
    source_features = get_features(model, source_path, mode, num_step)

  if target_feat.endswith('.pt'):
    target_features = torch.load(target_feat)
  else:
    target_features = get_features(model, target_path, mode, num_step)
  
  sim = similarity(source_features, target_features, 'dot')

  _, max_arg = torch.topk(sim, topk, dim=1, largest=False)

  for i, ip in enumerate(max_arg):
    si_name = source_name[i].split('/')[-1]

    dir_name = '{}/{}'.format(sv_path, i)
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
    shutil.copyfile(source_name[i], '{}/{}/SRC_{}'.format(sv_path, i, si_name))

    t_names = []
    for k in range(topk):
      ti_name = target_name[ip[k]].split('/')[-1]
      t_names.append(ti_name)

      shutil.copyfile(target_name[ip[k]], '{}/{}/{}'.format(sv_path, i, ti_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-feat', type=str, default='')
    parser.add_argument('--target-feat', type=str, default='')
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--gray', type=float, default=False)
    parser.add_argument('--num-step', type=int, default=100)
    opt = parser.parse_args()

    device = "cuda"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    if opt.gray:
      mode = 'bw'
    else:
      mode = 'rgb'

    # load model and scheduler
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    run_comparison(pipeline, opt.source_feat, opt.source, opt.target_feat, opt.target, opt.save, mode, opt.topk, opt.num_step)