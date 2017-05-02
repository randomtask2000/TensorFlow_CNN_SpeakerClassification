# Speaker Classification
Using convolutional networks on spectrograms to classify who is talking.
The goal of this project is to take 10 different voices and train a neural network to identify who is speaking from a set of new unlabeled voice samples. To measure the quality of the network is done by pure accuracy (correct attempts out of total attempts). The final attempt (convolutional networks on STFT images) achieves an accuracy of 0.939002.

# Libraries
All code is written in Python 3.5

Neural networks built with TensorFlow

Libraries needed for this project:

- librosa
- math
- matplotlib
- numpy
- pandas
- scipy
- skimage
- sklearn
- tensorflow
- time

# Data used
All data was created using files provided by LibriVox. All files used from LibriVox are in the public domain for free use.

Carl Manchester: https://librivox.org/the-911-commission-report-by-the-911-commission/

Sam Stinson: https://librivox.org/the-911-commission-report-by-the-911-commission/

Bill Boerst: https://librivox.org/abc-of-vegetable-gardening-by-eben-eugene-rexford/

Kandice Stehlik: https://librivox.org/across-the-years-by-eleanor-h-porter/

Julia Niedermaier: https://librivox.org/across-the-years-by-eleanor-h-porter/

Inah Derby: https://librivox.org/across-the-years-by-eleanor-h-porter/

Gabriela Cowan: https://librivox.org/across-the-years-by-eleanor-h-porter/

Tara Dow: https://librivox.org/across-the-years-by-eleanor-h-porter/

Don Jenkins: https://librivox.org/adventures-of-bindle-by-herbert-jenkins/

John Lieder: https://librivox.org/the-adventures-of-buster-bear-by-thornton-w-burgess/

The files downloaded are .mp3 files, which were first converted to .wav using VLC Media Player. The remaining preprocessing was done in Python.

The reduced 3-minute audio clips can be found in data.zip.

The spectrograms (images) used in the final two networks can be found in plt-spectrograms.zip and spectrograms.zip.

The trained neural networks are included as well as the following files:

- train_model_01.ckpt.data-00000-of-00001
- train_model_01.ckpt.index
- train_model_01.ckpt.meta
- train_model_02.ckpt.data-00000-of-00001
- train_model_02.ckpt.index
- train_model_02.ckpt.meta
- train_model_03.ckpt.data-00000-of-00001
- train_model_03.ckpt.index
- train_model_03.ckpt.meta
