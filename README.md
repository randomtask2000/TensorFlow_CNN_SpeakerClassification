# Speaker Classification
For a detailed writeup on this project, read the included pdf. 

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

![image](https://cloud.githubusercontent.com/assets/24555661/25774421/89eea6de-324b-11e7-9244-8c2be306542b.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25774427/a0de390e-324b-11e7-87fe-ffccb336b1b5.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25774552/5a8ec27c-324e-11e7-9b5a-8546259c9464.png)

All data was created using files provided by LibriVox. All files used from LibriVox are in the public domain for free use.

The datasets built for this project were made specifically for this project. The code to create the dataset can be found in preprocess.py. Below are the original audio files which are used to create the datasets. 3 minute versions of these files can be found in the data folder.

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

The reduced 3-minute audio clips can be found in the data folder. Each audio file has been zipped, so to use them in the Python code, unzip each file and place them in the data directory.

The spectrograms (images) used in the final two networks can be found in plt-spectrograms.zip and spectrograms.zip.

Unzip the folders in folder plt-spectrograms and folder spectrograms to use the images.

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
