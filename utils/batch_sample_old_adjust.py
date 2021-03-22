import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class Triplet(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(Sampler):
    """
    BatchSampler - from a ImageFloderLoader dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples): 
           
        self.labels = torch.LongTensor([item[1] for item in dataset.imgs])
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):

        self.count = 0
        indices = []

        while self.count + self.batch_size < len(self.dataset): # smple a batch            
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) 

            for class_ in classes:

                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    label_to_indices = np.random.choice(list(self.label_to_indices[class_]), self.n_samples, replace=True)
                    indices.extend(label_to_indices)
                else:
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0  
                                                 
            self.count += self.n_classes * self.n_samples

        return iter(indices)  
        """
        indices = []
        used_class = []

        while len(used_class) < len(self.labels_set):  # there are still imaged not used for train

            # choose those class are not used out
            labels_set = [label for label in self.labels_set if label not in used_class]

            # if there is not enough class to make the final batch, use the image used before
            if len(labels_set) < self.n_classes:
                labels_set = labels_set + list(np.random.choice(used_class, self.n_classes - len(labels_set), replace=False))

            classes = np.random.choice(labels_set, self.n_classes, replace=False)

            for class_ in classes:

                # there are not enough image make a batch
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):

                    # mark this class has been used out
                    used_class.append(class_)

                    # if there is not enough image for make the first batch
                    if self.used_label_indices_count[class_] == 0:
                        label_to_indices = list(np.random.choice(list(self.label_to_indices[class_]), self.n_samples, replace=True))
                    else:
                    # use those image used before to make the final batch
                        label_to_indices = list(self.label_to_indices[class_][self.used_label_indices_count[class_]:len(self.label_to_indices[class_])])
                        label_to_indices += list(np.random.choice(list(self.label_to_indices[class_])[0:self.used_label_indices_count[class_]],
                                                         self.n_samples - len(label_to_indices), replace=False))

                    label_to_indices = np.array(label_to_indices)
                    indices.extend(label_to_indices)

                else:
                    # choose image to make batch
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                    self.used_label_indices_count[class_] += self.n_samples

        return iter(indices)
        """
def __len__(self):
        return len(self.dataset) // self.batch_size
