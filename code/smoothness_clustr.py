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
import torch.nn as nn
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base-classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=0, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()




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
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        self.std = torch.tensor( [0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)

    def forward(self, x):
        _, embeddings = self.model((x-self.mean)/self.std)
        scores = get_softmax_probs(embeddings, self.magnet_data, 
            return_scores=True)
        return scores


def get_softmax_probs(embeddings, magnet_data, return_scores=False):

    device = embeddings.device
    num_clusters = magnet_data['cluster_classes'].size(0)
    num_classes = num_clusters // magnet_data['K']
    batch_size = embeddings.size(0)
    # distances states, for each instance, the distance to each cluster
    # Compute squared distances
    sq_distances = torch.cdist(
        embeddings, magnet_data['cluster_centers'], p=2)**2
    # Scale distances with variances
    sq_distances = sq_distances / magnet_data['variance'].unsqueeze(0)
    # Compute probs
    scores = torch.exp(-0.5 * sq_distances)
    largest_scores, indices = torch.topk(scores, k=magnet_data['L'], 
        largest=True, sorted=False)
    # Reshape to sum across clusters of the same class
    # this is of size [batch_size, num_classes, num_clusters]
    scores = scores.view(batch_size, num_classes, magnet_data['K'])
    # Perform sum (in the cluster dimension)
    scores = scores.sum(dim=2)
    # Normalizing factors
    # Get top-L CLOSEST (highest probabilities) clusters (need not be sorted)
    if magnet_data['normalize_probs']:
        labs_clusters = magnet_data['cluster_classes'].unsqueeze(0)
        labs_clusters = labs_clusters.expand(batch_size, num_clusters)
        labs_clusters = torch.take(labs_clusters, indices)
        scores = torch.zeros(batch_size, num_classes, device=device)
        unique_labels = torch.unique(labs_clusters)
        for label in unique_labels:
            label_mask = labs_clusters == label
            scores[:, label] = (largest_scores * label_mask).sum(dim=1)

    # Normalize probabilities in the class dimension (rows)
    to_return = scores if return_scores else F.normalize(scores, p=1, dim=1)
    return to_return

def acc(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    for img, label in dataloader:
        out = model(img.to(device))
        correct += (out.argmax(1).cpu() == label).sum()
    print('accuracy is {}'.format(float(correct)*100/len(dataset)))
    return correct/len(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
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
    _ = acc(base_classifier, dataset)
    for i in tqdm(range(len(dataset))):

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



