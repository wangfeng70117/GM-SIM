import os
import numpy as np
import torch
import argparse
from encoder_decoder import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)

    parser.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64, 32, 16, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512])
    args = parser.parse_args()

    dataset_path = args.dataset_path
    print(dataset_path)
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    ckpt_path = os.path.join(dataset_path, 'model_ckpt', 'best_ckpt.pth')
    data_path = os.path.join(dataset_path, 'clip_features.npy')
    output_path = os.path.join(dataset_path, 'clip_features3.npy')

    checkpoint = torch.load(ckpt_path)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to('cuda:0')
    model.load_state_dict(checkpoint)
    model.eval()
    features = torch.from_numpy(np.load(data_path)).to('cuda:0').float()
    with torch.no_grad():
        outputs = model.encode(features).to('cpu').numpy()
    print(f'outputs.shape is {outputs.shape}')
    np.save(output_path, outputs)
    print(f'clip features3 has saved in {output_path}')