# AuralisationCNN
This repo is for an example of auralisastion of CNNs that is demonstrated on ISMIR 2015.

Please visit my blog posting (http://keunwoochoi.blogspot.co.uk/2015/10/ismir-2015-lbd.html) for more information. Paper/poster/audio files are there!

# Usage
Load weights that you want to auralise. I'm using this function
```W = load_weights()```
to load my keras model, it can be anything else.
`W` is a list of weights for the convnet. (TODO: more details)

Then load source files, get STFT of it. I'm using `librosa`.

Then deconve it with `get_deconve_mask`.


