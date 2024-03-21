"""
Demo for neural network training with DD, tPD, and CD strategies

DD = Dataset-driven
tPD = Trained physics-driven
CD = Co-driven

Reference: Pending

@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import torch
from tqdm import tqdm
from optparse import OptionParser
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU

from network import UNet
from data_read import get_dataloaders
from train_func import train_net_CD, train_net_PD, train_net_DD

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--batch size', dest='batch_size', default=16, type='int', help='batch size')
    parser.add_option('--learning rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('--root', dest='root', default="./", help='root directory')
    parser.add_option('--input', dest='input', default='datasets/train_in/', help='folder of input')
    parser.add_option('--ground truth', dest='gt', default='datasets/train_gt/', help='folder of ground truth')
    parser.add_option('--model', dest='model', default='models_and_results/model_weights/', help='folder for model/weights')
    parser.add_option('-s', "--training strategy", dest='strategy', default="DD", choices=["DD", "tPD", "CD"], help='training strategy')

    (options, args) = parser.parse_args()
    return options

' Run of the training '
def setup_and_run(dir_input, dir_gt, dir_model, batch_size, epochs, lr, strategy):
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
            train_loss, input, output, gt = train_net_PD(net, device, train_loader, optimizer, loss_f, prop_dis=20,
                                                         norm=False, rand_to_holo=False)
        elif strategy == "CD":
            train_loss, input, output, gt = train_net_CD(net, device, train_loader, optimizer, loss_f, prop_dis=20,
                                                         norm=False, rand_to_holo=False)
        else:
            raise Exception("invalid training strategy")

        print('Training loss=', round(train_loss, 6))

        # see results in training
        if epoch % 10 == 0:
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
            strategy=args.strategy)