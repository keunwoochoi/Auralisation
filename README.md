# AuralisationCNN
This repo is for an example of auralisastion of CNNs that is demonstrated on ISMIR 2015.

Please visit my blog posting (http://keunwoochoi.blogspot.co.uk/2015/10/ismir-2015-lbd.html) for more information. Paper/poster/audio files are there!

# Files
auralise.py: includes all required function for deconvolution.
example.py: includes the whole code - just clone and run it by `python example.py`

# Folders
src_songs: includes three songs that I used in my blog posting.
src_templates: includes wave files that are designed to check what would happen for different chords and rhythms.
  ..,where the format of file name: [key]_[instrument_name]_[chord1]_[chord2]
    key: Eb, Bb, A3, G4
    instruments: EP, Piano, Acoustic guitar, Electric guitar (with over-drive), Saxophone, Strings
    chord: intervals, M, m, sus4, M7, 7, m7, dim

# Usage
Load weights that you want to auralise. I'm using this function
```W = load_weights()```
to load my keras model, it can be anything else.
`W` is a list of weights for the convnet. (TODO: more details)

Then load source files, get STFT of it. I'm using `librosa`.

Then deconve it with `get_deconve_mask`.


