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
from models import resnet
from var_estimator_network import FCNetwork

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()



class WrappedModel(nn.Module):
    def __init__(self, var_estimator_model, base_classifier, model_name):
        super(WrappedModel, self).__init__()

        self.var_estimator_model = var_estimator_model
        self.base_classifier = base_classifier
        self.model_name = model_name
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device)
        self.std = torch.tensor( [0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
    def split_base_classifier(self):
        if self.model_name == 'resnet18':
            a = nn.Sequential(self.base_classifier.conv1).eval()
            b = nn.Sequential(*list(self.base_classifier.children())[2:]).eval()
            return a, b
        else:
            raise NotImplementedError("Not implemented first conv layer extraction for this network")      
    def forward(self, x): #randomized smoothing do not use normalization in their call to the dataset.
        return self.base_classifier((x-self.mean)/self.std)


def get_model(path):
    var_estimator_model = FCNetwork(input_sz=img_sz,
                                    num_hidden_layers=1, num_hidden_nodes=50, args=args)
    base_classifier = resnet.resnet18(num_classes=10)
    model = WrappedModel(var_estimator_model, base_classifier, args.arch).to(device)


if __name__ == "__main__":
    # load the base classifier
    model = get_model(path)
    # create the smooothed classifier g
    # smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

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
        #Smooth the classifier with this sigma
        _, logvars = model.var_estimator_model(batch)
        args.sigma = torch.exp(0.5*logvars).item()
        smoothed_classifier = Smooth(model, get_num_classes(args.dataset),args.sigma)
        #Now you can use the same exac
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
