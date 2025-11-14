import os
import numpy as np
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo running in gpu $CUDA_VISIBLE_DEVICES')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataLoader import Autoencoder_dataset
from model import Autoencoder
import argparse


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64, 32, 16, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    data_dir = f'{dataset_path}/clip_features.npy'
    save_dir = f'{dataset_path}/model_ckpt'
    os.makedirs(save_dir, exist_ok=True)

    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = Autoencoder_dataset(data_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    best_eval_loss = 100.0
    best_epoch = 0

    test_epoch = num_epochs - num_epochs / 100
    print(f'训练到{test_epoch}之后停止迭代，开始选择最好的epoch')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for id, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)

            l2loss = l2_loss(outputs, data)
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录当前 batch 的损失

            # 更新进度条的后缀显示
        if epoch % 50 == 0:
            tqdm.write(
                f"Batch {id + 1}/{len(train_loader)}: L2 Loss: {l2loss.item():.8f}, Cosine Loss: {cosloss.item():.8f}, Total Loss: {loss.item():.8f}")
        if epoch > test_epoch:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0")
                with torch.no_grad():
                    outputs = model(data)
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                print(f'model has save in {save_dir}/best_ckpt.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_ckpt.pth'))

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch}_ckpt.pth'))

    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))