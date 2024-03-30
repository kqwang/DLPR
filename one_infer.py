"""
Neural network inference with DD, tPD, and CD strategies

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
import os.path
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from skimage.metrics import mean_squared_error as compare_mse

from functions.network import UNet
from functions.data_read import get_dataloader_for_test

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--result', dest='result', default="models_and_results/results_DD_tPD_CD/", help='folder for inference results')
    parser.add_option('-r', '--root', dest='root', default="./", help='root directory')
    parser.add_option('-m', '--model', dest='model', default='models_and_results/model_weights/', help='folder for model/weights')
    parser.add_option('-i', '--input', dest='input', default="datasets/test_in/", help='folder of input')
    parser.add_option('-g', '--gt', dest='gt', default="datasets/test_gt/", help='folder of ground truth')
    parser.add_option('-s', "--inference strategy", dest='strategy', default="DD", choices=["DD", "tPD", "CD"], help='inference strategy, DD, tPD, or CD')
    (options, args) = parser.parse_args()
    return options

' Run of the inference '
def setup_and_run(load_weights, dir_input, dir_gt, resultdir, strategy):

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Generate a folder for model weights and network results
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    # Create the model
    net = UNet().to(device)
    net.eval()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

    # Load old weights
    if strategy == 'DD':
        checkpoint = torch.load(load_weights + "DD_weights.pth", map_location='cpu')
    elif strategy == 'tPD':
        checkpoint = torch.load(load_weights + "tPD_weights.pth", map_location='cpu')
    elif strategy == 'CD':
        checkpoint = torch.load(load_weights + "CD_weights.pth", map_location='cpu')
    else:
        raise Exception("invalid inference strategy")
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    # Load the dataset
    loader = get_dataloader_for_test(dir_input, dir_gt)

    # Set the header for csv
    header = ['No.', 'mse']


    with torch.no_grad():
        for (input, gt) in loader:
            input, gt = input.to(device), gt.to(device)
            for th in range(0, len(input)):
                output = net(torch.unsqueeze(input[th], 0))

                input_numpy = input[th].squeeze(0).cpu().numpy()
                output_numpy = output.squeeze(0).squeeze(0).cpu().numpy()
                output_numpy = output_numpy - np.min(output_numpy)
                gt_numpy = gt[th].squeeze(0).cpu().numpy()

                mse = compare_mse(output_numpy, gt_numpy)


                # Set the values for csv
                values = [th + 1, mse]
                print(values)
                # Save MSE to a csv

                path_csv = resultdir + strategy + "_MSE" + ".csv"
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
                filename_plt = resultdir + strategy + '_sample' + (str(th + 1).zfill(6)) + '_results.png'
                plt.savefig(filename_plt)
                # plt.show()

                filename = resultdir + strategy + '_sample' + (str(th + 1).zfill(6)) + '_results.mat'
                sio.savemat(filename, {'input': input_numpy, 'output': output_numpy, 'gt': gt_numpy})

' Run the application '
if __name__ == '__main__':
    args = get_args()
    setup_and_run(load_weights=args.root + args.model,
            dir_input=args.root +args.input,
            dir_gt=args.root + args.gt,
            resultdir=args.root + args.result,
            strategy=args.strategy)