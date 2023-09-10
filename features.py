import torch
import os
from PIL import Image, ImageEnhance
import argparse
from combine import combine_features

def load_img(path, size, mode):
    low_res_img = Image.open(path).convert('RGB')
    if mode == 'bw':
        filter_bw = ImageEnhance.Color(low_res_img)
        img = filter_bw.enhance(0)
    else:
        img = low_res_img

    img = img.resize(size)
    return img

def get_features(model, img_path, mode, num_step=100):
    features = []
    for fRoot, fDirs, fFiles in os.walk(img_path):
        for ffile in fFiles:
            full_path = os.path.join(fRoot, ffile)

            size = (128, 128)
            low_res = load_img(full_path, size , mode)
            latent_emb = model(low_res, num_inference_steps=num_step, eta=1)
            emb = get_feature_vector(latent_emb, 'raw')
            feat = torch.unsqueeze(emb.detach().cpu(), dim=0)
            features.append(feat)

    features = torch.cat(features, dim=0)
    return features

def get_feature_vector(latent_in, mode):
    if mode == 'max':
        pool = torch.nn.MaxPool2d(4, padding=2, stride=2)
        output = pool(latent_in)
    elif mode == 'avg':
        pool = torch.nn.AvgPool2d(4, padding=2, stride=2)
        output = pool(latent_in)
    else:
        output = latent_in
    fv = torch.flatten(output).detach().clone()
    return fv

def save_features(model, img_path, sv_path, resume, mode, num_step=100):
    i = resume
    size = (128, 128)
    all_path = []
    for fRoot, fDirs, fFiles in os.walk(img_path):
        for ffile in fFiles:
            full_path = os.path.join(fRoot, ffile)#.replace('/', '\\')
            all_path.append(full_path)

    all_path = all_path[i:]

    for p in all_path:               
        low_res = load_img(p, size, mode)
        latent_emb = model(low_res, num_inference_steps=num_step, eta=1)
        emb = get_feature_vector(latent_emb, 'none')
        emb = emb.detach().cpu()
        full_sv_path = sv_path + '/' + str(i) + '.pt'
        torch.save(emb, full_sv_path)
        i += 1
    
    combine_features(sv_path, sv_path)
            

#Run to save image features
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--num-step', type=int, default=100)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--gray', type=float, default=False)
    opt = parser.parse_args()

    device = "cuda"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    # load model and scheduler
    from SR_pipe import LDMSuperResolutionPipeline
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    if opt.gray:
        mode = 'bw'
    else:
        mode = 'rgb'

    #create save directory if it doesn't exist
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    save_features(pipeline, opt.source, opt.save, opt.resume, mode, opt.num_step)
