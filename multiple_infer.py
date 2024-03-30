"""
Neural network inference with uPD and tPDr strategies

uPD = untrained physics-driven
tPDr = Trained physics-driven with refinement

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify GPU
import torch
import numpy as np
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from optparse import OptionParser
from skimage.metrics import mean_squared_error as compare_mse

from functions.network import UNet
from functions.data_read import get_dataloaders_th
from functions.train_func import train_net_PD

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('--result', dest='result',
                      default="models_and_results/results_uPD_tPDr/", help='folder for inference results')
    parser.add_option('--cycles', dest='cycles', default=1000, type='int', help='number of cycles')
    parser.add_option('--learning rate', dest='lr', default=0.001, type='float', help='learning rate')
    parser.add_option('--root', dest='root', default="./", help='root directory')
    parser.add_option('--input', dest='input', default='datasets/test_in/', help='folder of input')
    parser.add_option('--ground truth', dest='gt', default='datasets/test_gt/', help='folder of ground truth')
    parser.add_option('--model', dest='model', default='models_and_results/model_weights/', help='folder for model weights')
    parser.add_option('--prop dis', dest='prop_dis', default=20, type='float', help='propagation distance, mm')
    parser.add_option('--rand to holograms', dest='rth', default=True, type='int', help='Add rand noise matrix to hologram or not?')
    parser.add_option('--dimension', dest='dim', default=256, type='int', help='dimension of the dataset, [dim, dim]')
    parser.add_option('--norm', dest='norm', default=False, type='int', help='Set True for holograms in the range of [0,1]')
    parser.add_option('-s', "--inference strategy", dest='strategy', default="uPD", choices=["uPD", "tPDr"], help='inference strategy, uPD or tPDr')
    (options, args) = parser.parse_args()
    return options

' Run of the inference '
def setup_and_run(dir_input, dir_gt, dir_model, cycles, resultdir, lr, prop_dis, rth, dim, norm, strategy):
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Generate a folder for model weights and network results
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    header = ['No.', 'Cycle', 'Learning rate', 'MSE']  # Set the header for csv
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

        train_loader = get_dataloaders_th(dir_input, dir_gt, th)  # dataset loader
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)  # Definition of the optimizer
        loss_f = torch.nn.L1Loss()  # Definition of the loss function

        # Ready to use the tqdm
        for cycle in tqdm(range(cycles)):
            if args.lr > 0.00001:
                if cycle % 10 == 0:
                    args.lr = args.lr * 0.95
            print('\nLearning rate=', round(args.lr, 6))

            # Get training loss function and validating loss function
            train_loss, input, output, gt = train_net_PD(net, device, train_loader, optimizer, loss_f, prop_dis=prop_dis, norm=norm, rand_to_holo=rth, dim=dim)
            print('Training loss=', round(train_loss, 6))


            if (cycle == 0) or ((cycle+1) % 100) == 0:
                # input, output, gt to numpy
                input_numpy = input.squeeze(0).squeeze(0).cpu().numpy()
                output_numpy = output.squeeze(0).squeeze(0).detach().cpu().numpy()
                output_numpy = output_numpy - np.min(output_numpy)
                gt_numpy = gt.squeeze(0).squeeze(0).cpu().numpy()

                # see results in cycle inference
                plt.subplot(131)
                plt.imshow(input_numpy, cmap='gray')
                plt.title('input (hologram)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(output_numpy, cmap="inferno")
                plt.title('output (phase)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(gt_numpy, cmap="inferno")
                plt.title('gt (phase)')
                plt.colorbar(fraction=0.05, pad=0.05)
                plt.axis('off')
                filename_plt = resultdir + strategy + '_sample' + (str(th + 1).zfill(3)) + '_cycle' + (str(cycle + 1).zfill(6)) + '_results.png'
                plt.savefig(filename_plt)
                plt.show()

                # save results in .mat
                filename = resultdir + strategy + '_sample' + (str(th + 1).zfill(3)) + '_cycle' + (str(cycle + 1).zfill(6)) + '_results.mat'
                sio.savemat(filename, {'input': input_numpy, 'output': output_numpy, 'gt': gt_numpy})

                # save lr and mse in .csv
                mse = compare_mse(gt_numpy, output_numpy)
                values = [th + 1, cycle + 1, round(args.lr, 6), mse]
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
        cycles=args.cycles,
        resultdir=args.root + args.result,
        lr=args.lr,
        prop_dis=args.prop_dis,
        rth=args.rth,
        dim=args.dim,
        norm=args.norm,
        strategy=args.strategy)