import torch
import os
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import argparse

def load_img(path, size, color):
    low_res_img = Image.open(path).convert('RGB')
    if color == 'bw':
        filter_bw = ImageEnhance.Color(low_res_img)
        img = filter_bw.enhance(0)
    else:
        img = low_res_img

    img = img.resize(size)
    return img

def get_batch(path_list, batch_size, color): 
    size = (128,128)
    transform = transforms.ToTensor()

    img_batch = []
    for p in path_list:
        single = load_img(p, size, color)
        img_batch.append(transform(single))
        if len(img_batch) == batch_size or p == path_list[-1]:
            yield torch.stack(img_batch)
            img_batch = []
    
def get_features(model, img_path, color, batch_size, num_step=100):
    features = []
    all_path = []

    for fRoot, fDirs, fFiles in os.walk(img_path):
        for ffile in fFiles:
            full_path = os.path.join(fRoot, ffile).replace('/', os.sep)
            all_path.append(full_path)

    for batch in get_batch(all_path, batch_size, color):
        latent_embs = model(batch, num_inference_steps=num_step, eta=1)
        lat_matrix = latent_embs.detach().cpu()
        features.append(lat_matrix)

    return features

def save_features(model, img_path, sv_path, resume, color, batch_size, num_step=100):
    i = resume
    size = (128, 128)
    all_path = []
    for fRoot, fDirs, fFiles in os.walk(img_path):
        for ffile in fFiles:
            full_path = os.path.join(fRoot, ffile).replace('/', os.sep)
            all_path.append(full_path)

    all_path = all_path[i:]

    for batch in get_batch(all_path, batch_size, color):
        latent_embs = model(batch, num_inference_steps=num_step, eta=1)
        lat_matrix = latent_embs.detach().cpu()

        for b in range(len(batch)):
            full_sv_path = sv_path + '/' + str(i) + '.pt'
            torch.save(lat_matrix[b].clone(), full_sv_path)
            i += 1
    
#Run to save image features
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--num-step', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--gray', action='store_true')
    opt = parser.parse_args()

    if not opt.source or not opt.save:
        raise Exception("No source folder or save path provided")

    device = "cuda"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    #create save directory if it doesn't exist
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    

    # load model and scheduler
    from SR_pipe import LDMSuperResolutionPipeline
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    if opt.gray:
        mode = 'bw'
    else:
        mode = 'rgb'

    save_features(pipeline, opt.source, opt.save, opt.resume, mode, opt.batch_size, opt.num_step)
