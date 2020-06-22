# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from resnet import ResNet18

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base-classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the base classifier
    # checkpoint = torch.load(args.base_classifier)
    # base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    # base_classifier.load_state_dict(checkpoint['state_dict'])

    #Load our model and make the wrapper
    model = ResNet18(num_classes=10).to(device)
    base_classifier = load_model(model, checkpoint=args.base_classifier, L=10, 
        normalize_probs=False)
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()




def load_model(model, checkpoint, L=None, normalize_probs=None):
    # Load the checkpoint
    if not checkpoint.endswith('.pth'):
        print('Checkpoint isnt ".pth" file: inferring model_best.pth')
        ckpt = torch.load(osp.join(checkpoint, 'model_best.pth'))
    else:
        ckpt = torch.load(checkpoint)
    # Construct magnet data
    try:
        magnet_data = {
            'cluster_classes'   : ckpt['cluster_classes'],
            'cluster_centers'   : ckpt['cluster_centers'],
            'variance'          : ckpt['variance'],
            'L'                 : ckpt['L'] if L is None else L,
            'K'                 : ckpt['K'],
            'normalize_probs'   : ckpt['normalize_probs'] if normalize_probs is None else normalize_probs
        }
        print('Succesfully loaded magnet_data from checkpoint')
    except:
        magnet_data = None
        print('Unable to load magnet_data from checkpoint. '
            'Regular training is inferred')

    # Load and prepare the model
    try:
        state_dict = ckpt['state_dict']
    except:
        print(f'Checkpoint "{checkpoint}" does not contain a dict. Assuming '
            f'it is the state_dict itself (like TRADES models)')
        state_dict = { k.replace('model.', '') : v for k, v in ckpt.items() }
    
    model.load_state_dict(state_dict)
    # mean = distrib_params['mean'].view(1,3,1,1).to(device)
    # std = distrib_params['std'].view(1,3,1,1).to(device)
    model_wrapper = MagnetModelWrapper(model, magnet_data, None, None)
    model_wrapper.eval()
    model_wrapper.float()
    return model_wrapper

class MagnetModelWrapper(nn.Module):
    def __init__(self, model, magnet_data, mean, std):
        super(MagnetModelWrapper, self).__init__()
        self.model = model
        self.magnet_data = magnet_data
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.magnet_data is None:
            # If TRADES model -> renormalize data
            x = x * self.std + self.mean if self.mean is not None else x
            scores, _ = self.model(x)
        else:
            _, embeddings = self.model(x)
            scores = get_softmax_probs(embeddings, self.magnet_data, 
                return_scores=True)
        return scores
