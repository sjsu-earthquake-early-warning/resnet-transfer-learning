"""TODO: Fill this out."""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from skimage import transform, color
import numpy as np
import scipy.fftpack
from scipy.signal import spectrogram
import h5py
import matplotlib.pyplot as plt

# original_train_file = 'data/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5'
train_file = 'data/scsn_p_2000_2017_6sec_0.5r_pick_train_mix.hdf5'
test_file = 'data/scsn_p_2000_2017_6sec_0.5r_pick_test_mix.hdf5'
dataset_size = 1000
features_key = 'X'
labels_key = 'pwave'

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WaveformDataset(Dataset):
  """Has p-wave and noise waveforms."""

  def __init__(self, filepath, dataset_size=-1, transform=None):
    """
    Args:
      filepath (string): Path to .h5py file.
      transform (callable, optional): Applied on the dataset.
    """
    waveforms_key = 'X'
    labels_key = 'pwave' 

    # Open, read, and close the file.
    file = h5py.File(filepath, 'r')
    dataset_size = dataset_size if dataset_size > 0 \
                   else len(file[waveforms_key])
    self.samples = {
      "waveforms": file[waveforms_key][:dataset_size],
      "labels": file[labels_key][:dataset_size]
    }
    file.close()

    # Note: transform is the argument passed to __init__, NOT skimage.transform
    self.transform = transform

  def __len__(self):
    return len(self.samples["waveforms"])

  def __getitem__(self, idx):
    waveform = self.samples["waveforms"][idx]
    if self.transform:
      waveform = self.transform(waveform)
    return {
      "waveform": waveform,
      "label": self.samples["labels"][idx]
    }

class Spectrogram(object):
  """Obtain the spectrogram of a given waveform."""

  def __call__(self, waveform):
    freqs, times, Sx = spectrogram(waveform, fs=100, window='hamming',
                                   nperseg=30, noverlap=0,
                                   detrend='linear', scaling='spectrum')
    return Sx

def main():
  train_filepath = "data/scsn_p_2000_2017_6sec_0.5r_pick_train_mix.hdf5"
  test_filepath = "data/scsn_p_2000_2017_6sec_0.5r_pick_test_mix.hdf5"
  dataset_size = 5000

  t = transforms.Compose([
    Spectrogram(),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
  ])

  train_dataset = WaveformDataset(filepath=train_filepath,
                                  dataset_size=dataset_size,
                                  transform=t)
  test_dataset = WaveformDataset(filepath=test_filepath,
                                 dataset_size=dataset_size,
                                 transform=t)

  train_dataloader = DataLoader(train_dataset,
                                batch_size=16,
                                shuffle=True)
  test_dataloader = DataLoader(test_dataset,
                               batch_size=16,
                               shuffle=True)

  

if __name__ == "__main__":
  main()