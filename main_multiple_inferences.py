"""
Neural network inference with uPD and tPDr strategies

uPD = untrained physics-driven
tPDr = Trained physics-driven with refinement

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import torch
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
import csv
import scipy.io as sio
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU

from skimage.metrics import mean_squared_error as compare_mse

from network import UNet
from data_read import get_dataloaders_th
from train_func import train_net_PD

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('--result', dest='result',
                      default="models_and_results/results_uPD_tPDr/", help='folder for inference results')
    parser.add_option('--epochs', dest='epochs', default=10000, type='int', help='number of epochs')
    parser.add_option('--learning rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('--root', dest='root', default="./", help='root directory')
    parser.add_option('--input', dest='input', default='datasets/test_in/', help='folder of input')
    parser.add_option('--ground truth', dest='gt', default='datasets/test_gt/', help='folder of ground truth')
    parser.add_option('--model', dest='model', default='models_and_results/model_weights/', help='folder for model weights')
    parser.add_option('-s', "--inference strategy", dest='strategy', default="uPD", choices=["uPD", "tPDr"], help='inference strategy, uPD or tPDr')

    (options, args) = parser.parse_args()
    return options

' Run of the inference '
def setup_and_run(dir_input, dir_gt, dir_model, epochs, resultdir, lr, strategy):
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Generate a folder for model weights and network results
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    header = ['No.', 'Epoch', 'Learning rate', 'MSE']  # Set the header for csv
    ids = [f[:-4] for f in os.listdir(dir_input)]  # Read the names of the images

    # inference for th sample
    for th in range(len(ids)):
        # Create the model
        net = UNet().to(device)
        net.train()
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

        # pre-train or un-train
        if strategy == 'tPDr':
            print(['Please make sure there are pretrained neural network weights available in' + dir_model])
            path_checkpoint = dir_model + '/tPD_weights.pth'
            checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
            net.load_state_dict(checkpoint['state_dict'])
        elif strategy == 'uPD':
            pass
        else:
            raise Exception("invalid inference strategy")

        # dataset loader
        train_loader = get_dataloaders_th(dir_input, dir_gt, th)

        # Definition of the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)

        # Definition of the loss function
        loss_f = torch.nn.L1Loss()

        # Ready to use the tqdm
        for epoch in tqdm(range(epochs)):
            if args.lr > 0.00001:
                if epoch % 10 == 0:
                    args.lr = args.lr * 0.99
            print('\nLearning rate=', round(args.lr, 6))

            # Get training loss function and validating loss function
            train_loss, input, output, gt = train_net_PD(net, device, train_loader, optimizer, loss_f, prop_dis=20, norm=False, rand_to_holo=True)
            print('Training loss=', round(train_loss, 6))


            if epoch % 200 == 0:

                # see results in cycle inference
                plt.subplot(131)
                plt.imshow(input[0, 0, :, :].cpu().numpy(), cmap='gray')
                plt.title('input (hologram)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="inferno")
                plt.title('output (phase)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(gt[0, 0, :, :].detach().cpu().numpy(), cmap="inferno")
                plt.title('gt (phase)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                filename_plt = resultdir + strategy + '_sample' + (str(th + 1).zfill(3)) + '_cycle' + (str(epoch + 1).zfill(6)) + '_results.png'
                plt.savefig(filename_plt)
                plt.show()

                # save results in .mat and save loss in .csv
                input_numpy = input.squeeze(0).squeeze(0).cpu().numpy()
                output_numpy = output.squeeze(0).squeeze(0).detach().cpu().numpy()
                output_numpy = output_numpy - np.min(output_numpy)
                gt_numpy = gt.squeeze(0).squeeze(0).cpu().numpy()
                mse = compare_mse(gt_numpy, output_numpy)
                values = [th+1, epoch + 1, round(args.lr, 6), mse]
                filename = resultdir + strategy + '_sample' + (str(th + 1).zfill(3)) + '_cycle' + (str(epoch + 1).zfill(6)) + '_results.mat'
                sio.savemat(filename, {'input': input_numpy, 'output': output_numpy, 'gt': gt_numpy})
                path_csv = resultdir + strategy + '_sample' + (str(th + 1).zfill(3)) + "_loss_and_others" + ".csv"
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

' Run the application '
if __name__ == "__main__":
    args = get_args()
    setup_and_run(
            dir_input=args.root + args.input,
            dir_gt=args.root + args.gt,
            dir_model=args.root + args.model,
            epochs=args.epochs,
            resultdir=args.root + args.result,
            lr=args.lr,
            strategy=args.strategy)