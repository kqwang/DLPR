"""
Neural network training with DD, tPD, and CD strategies

DD = Dataset-driven
tPD = Trained physics-driven
CD = Co-driven

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from optparse import OptionParser

from functions.network import UNet
from functions.data_read import get_dataloaders
from functions.train_func import train_net_CD, train_net_PD, train_net_DD

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--batch size', dest='batch_size', default=16, type='int', help='batch size')
    parser.add_option('--learning rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('--root', dest='root', default="./", help='root directory')
    parser.add_option('--input', dest='input', default='datasets/train_in/', help='folder of input')
    parser.add_option('--ground truth', dest='gt', default='datasets/train_gt/', help='folder of ground truth')
    parser.add_option('--model', dest='model', default='models_and_results/model_weights/', help='folder for model weights')
    parser.add_option('--rand to holograms', dest='rth', default=False, type='int', help='Add rand noise matrix to hologram or not?')
    parser.add_option('--prop dis', dest='prop_dis', default=20, type='float', help='propagation distance, mm')
    parser.add_option('--dimension', dest='dim', default=256, type='int', help='dimension of the dataset, [dim, dim]')
    parser.add_option('--norm', dest='norm', default=False, type='int', help='Set True for holograms in the range of [0,1]')
    parser.add_option('-s', "--training strategy", dest='strategy', default="DD", choices=["DD", "tPD", "CD"], help='training strategy, DD, tPD, and CD')
    parser.add_option('--alpha', dest='alpha', default=0.3, type='float', help='weight of dataset term in the loss function of CD')
    (options, args) = parser.parse_args()
    return options

' Run of the training '
def setup_and_run(dir_input, dir_gt, dir_model, batch_size, epochs, lr, rth, prop_dis, dim, norm, strategy, alpha):
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the model
    net = UNet().to(device)
    net.train()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

    # dataset loader
    train_loader = get_dataloaders(dir_input, dir_gt, batch_size)

    # Definition of the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.00)

    # Definition of the loss function
    loss_f = torch.nn.L1Loss()

    # Set the header for csv
    header = ['epoch', 'learning rate', 'train loss']

    # set the best loss for model weights saving
    best_loss = 1000000

    # Generate a folder for network training data
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    # Ready to use the tqdm
    for epoch in tqdm(range(epochs)):
        if args.lr > 0.00001:
            if epoch % 5 == 0:
                args.lr = args.lr * 0.95
        print('\nLearning rate=', round(args.lr, 6))

        # network training with DD, tPD, or CD
        if strategy == "DD":
            train_loss, input, output, gt = train_net_DD(net, device, train_loader, optimizer, loss_f)
        elif strategy == "tPD":
            train_loss, input, output, gt = train_net_PD(net, device, train_loader, optimizer, loss_f, prop_dis=prop_dis,
                                                         norm=norm, rand_to_holo=rth, dim=dim)
        elif strategy == "CD":
            train_loss, input, output, gt = train_net_CD(net, device, train_loader, optimizer, loss_f, prop_dis=prop_dis,
                                                         norm=norm, rand_to_holo=rth, dim=dim, alpha=alpha)
        else:
            raise Exception("invalid training strategy")

        print('Training loss=', round(train_loss, 6))

        # see results in training
        if (epoch == 0 or (epoch+1) % 10 == 0):
            plt.subplot(131)
            plt.imshow(input[0, 0, :, :].cpu().numpy(), cmap='gray')
            plt.title('input (hologram)')
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.axis('off')
            plt.subplot(132)
            output_np = output[0, 0, :, :].detach().cpu().numpy()
            output_np = output_np - np.min(output_np)
            plt.imshow(output_np, cmap="inferno")
            plt.title('output (phase)')
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(gt[0, 0, :, :].detach().cpu().numpy(), cmap="inferno")
            plt.title('gt (phase)')
            plt.colorbar(fraction=0.05, pad=0.05)
            plt.axis('off')

            filename_plt = dir_model + strategy + '_' + (str(epoch + 1).zfill(6)) + '_in_out_gt.png'
            plt.savefig(filename_plt)
            plt.show()

        # Set the values for csv
        values = [epoch+1, args.lr, train_loss]

        # Save epoch, learning rate, train loss, val loss and time cost now to a csv
        path_csv = dir_model + strategy + "_loss_and_others" + ".csv"
        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(values)
        file.close()

        # Save model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': train_loss,
                    'optimizer': optimizer.state_dict(),
                }, dir_model + strategy + "_weights" + ".pth")

' Run the application '
if __name__ == "__main__":
    args = get_args()
    setup_and_run(
        dir_input=args.root + args.input,
        dir_gt=args.root + args.gt,
        dir_model=args.root + args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        rth=args.rth,
        prop_dis=args.prop_dis,
        dim=args.dim,
        norm=args.norm,
        strategy=args.strategy,
        alpha=args.alpha)
