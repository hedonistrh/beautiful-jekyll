# Literature Review for Speaker Change Detection

**Draft version. So that, there can be many typos and unreferenced quote. I will add their result and dataset. In addition to that, I will add some papers. Feel free to send e-mail to me.**

Some useful links for dataset:
- https://voice.mozilla.org/en/data
- http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/

In general, a speaker diarization system consists of two main parts: segmentation and clustering. Segmentation aims to detect all speaker change points. The most widely used method is the Bayesian Information Criterion (BIC) based segmentation. More recently, researcher focus to using of Deep Learning. 

Speaker diarization is the task of determining “who spoke when” in an audio stream that usually contains an unknown amount of speech from an unknown number of speakers. Speaker change detection is an important part of speaker diarization systems. It aims at finding the boundaries between speech turns of two different speakers.


### 1) [_Multimodal Speaker Segmentation and Diarization using Lexical and Acoustic Cues via Sequence to Sequence Neural Networks_](https://arxiv.org/abs/1805.10731)


- _"In our work we propose a system that incorporates both lexical cues and acoustic cues to build a system closer to how humans employ information."_ They use the lexical information to improve results. If the script of recording is available, they directly use it. If not, ASR has been used to extract lexical cues.

- In this work, main architecture is **sequence to sequence(seq2seq)** which summarize the whole sequence into an embedding. Moreover, it can integrate information and process variable length sequences. Thus, this model can capture temporally encoded information from both before and after the speaker change points. Also, this model use the attention mechanism so that system can learns which information is most important to characterize the speaker.

![alt text](https://docs.google.com/uc?id=18O2dSgm4yrL3lmIuN74vUyTTiqgWS0gd)

![alt text](https://docs.google.com/uc?id=1tF4YKvAEEGkVZzDIeLnHeF4rF7ohIkLq)


- Encoder takes the sequence of word representation and MFCC (_13 dimensional. Extracted with a 25ms window and 10ms shift_. Decoder produces a sequence of word with speaker IDs. Thus, system can learn speaker change points.

    - Source sequence is 32 words (one hot word vector) which comes from reference script or ASR output. Target sequences is 32 words and added speaker turns tokens.

- To maximize the accuracy of speaker turn detection, they use the shift and overlap scheme to predict the speaker turn.

![alt text](https://docs.google.com/uc?id=1hQfGOxrrLgld1TjGsDtl94Ul27TNa8Xj)


### 2) [_Speaker2Vec: Unsupervised Learning and Adaptation of a Speaker Manifold using Deep Neural Networks with an Evaluation on Speaker Segmentation_](http://scuba.usc.edu/pdf/jati2017_Speaker2Vec.pdf)

Their aim is derive a speaker-charecteristic manifold learned in an **unsupervised** manner.

Note: _State-of-the-art unsupervised speaker segmentation approaches are based on measuring the statistical distance between two consecutive windows in audio signal. For instance: BIC, KL Divergence. These methods use some low level features like MFCC for signal parameterization_

- They assume that temporally-near speech segments belong to the same spekar so that such a joint representation connecting these nearby segments can encode their common information. Thus, this bottleneck representation will be capturing mainly speaker-spesific information. When test this system, simple distance metric is applied to detect speaker change points.

- Given any small segment (say 1 second) of speech, a trained Speaker2Vec model can find its latent “representation vector” or “embedding” which contains mostly speaker-specific information.

- They train a DNN on unlabeled data to learn a speaker-characteristics manifold, use the trained model to generate embeddings for the test audio, and use those embeddings to find the speaker change points.

Methodology of this work
 
- Tries to learn speaker characteristics manifold with autoencoder. 

    - They do not try to reconstruct input. They try ro reconstruct a small window of speech from a temporally nearby window. According to their hypothesis, given the two windows belong to same speaker. (As we guess, at some case, this assumption is not true. However, if we look at the rate of this situation, it will be negligible. Also, after the first training, they train the system on homogenous segments of speech.) With this reconstruction, the system can get rid of unnecessary features and capture the most common information between two window. Thus, it can learn speaker-characteristic manifold.

![alt text](https://docs.google.com/uc?id=1epse9ba1fRTdmyN3pF0XoECEHcaiLsa3)


- For the segmentation, system use the embeddings instead of original MFCC features. They use the asymmetric KL divergence for segmentation.

- They use two-pass algorithm.

    - Find the speaker change points by trained DNN model.
    - Get all possible speaker homogeneous regions.
    - Retrain the same DNN again on these homogeneous segments of speech.

**Experiment**

- They use data from [TED-LIUM](http://www.openslr.org/7/) and Youtube for training. To compare baseline methods, they use TED-LIUM evaluation data.

![alt text](https://docs.google.com/uc?id=1ka7SDKEY481IF1Vfd0lmVSXyTP2I8F8h)


- There are 2 different architecture. One for TED-LIUM, Youtube (_4000 - 2000 - 40 - 2000 - 4000_) and other one for YoutubeLarge. (_6000 - 2000 - 40 - 2000 - 6000 - 4000_) The embeddings layer is always 40. Because, they want to represent MFCC dimension with this layer.

- They compare their result with state-of-art methods on the artificially created TIMIT dataset.

![alt text](https://docs.google.com/uc?id=1bdaPmhY0pHH9W5WLpa1qj_y0yc6h4dvm)

### 3) [_TRISTOUNET: TRIPLET LOSS FOR SPEAKER TURN EMBEDDING_](https://arxiv.org/abs/1609.04301)

_"TristouNet is a neural network architecture based on Long Short-Term Memory recurrent networks, meant to project speech sequences into a fixed-dimensional euclidean space. Thanks to the triplet loss paradigm used for training, the resulting sequence embeddings can be compared directly with the euclidean distance, for speaker comparison purposes."_

![alt text](https://docs.google.com/uc?id=12K2cZ0dZohRJaLEkgXlWxkqq2G-ZwNqd)

This figure summarizes main idea. When train the system, system takes three different sequence (Anchor and positive belongs to same class and negative comes from different speaker.) and converts to embedding. After that, triplet loss is applied to these embeddings. The triplet loss functions's aim is that minimizes distance between embeddins of _anchor_ and _positive_ and maximize distance between embeddings of _anchor_ and _negative_. 

![alt text](https://docs.google.com/uc?id=1GfKoP_Olvnn_NkZRsJ_8NVIRSETIWo3S)

This figure depicts how embedding is created from sequence.  

### 4) [_Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks_](https://pdfs.semanticscholar.org/edff/b62b32ffcc2b5cc846e26375cb300fac9ecc.pdf)

Speaker change detection is like a binary seqeunce labelling task and addressed by Bidirectional long short term memory networks. (Bi-LSTM)

Previously, writers proposed _TristouNet_, at that system, euclidean distance is used. But, that system tend to miss boundaries in fast speaker interactions because of relatively long adjacent sliding windows. (2 seconds or more)

_Note: "In particular, our proposed approach is the direct translation of the work by Gelly et al. where they applied Bi-LSTMs on overlapping audio sequences to predict whether each frame corresponds to a speech region or a non-speech one."_ 
    - [Gelly et al.'s paper](ftp://tlp.limsi.fr/public/IS2015_vad.pdf)

![alt text](https://docs.google.com/uc?id=1ciFHkElqKFk6SDfXPs4U8rsFVbmiuuT1)

They use the MFCC which comes from overlapping slicing windows as input, output is binary class. System use the binary cross-entropy loss function to train.

![alt text](https://docs.google.com/uc?id=19XHlHsMwgslNmstqUXFAe7-HYusygX5S)

- Bi-LSTMs allow the process sequences in forward and backward directions, making use of both past and future contexts.

- To solve class imbalance, the number of positive labels is increased artificially by labelling as positive every frame in the direct neighborhood of the manually annotated change point. Positive neighboorhood of 100ms (50ms on both side) is used around each change point.

- Long audio sequences are split into short fixed-length overlapping sequences. These are 3.2s long with a step of 800 ms.

**Experiment**

- They use ETAPE TV subset. 
- MFCC as input. 
- Baselines are BIC, Gaussian Divergence (both of them use 2s adjacent windows) and TristouNet.


![alt text](https://docs.google.com/uc?id=1hbOjusrAIPDr6GM5fs5dVUMrjmnrUykT)

- _"We have developed a speaker change detection approach using bidirectional long short-term memory networks. Experimental results on the ETAPE dataset led to significant improvements over conventional methods (e.g., based on Gaussian divergence) and recent state-of-the-art results based on TristouNet embeddings."_

### 5) [_Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings_](https://arxiv.org/abs/1708.02840)

They are trying to solve speaker diarization problem via 2-step approach.

- To classify speaker, train a NN in a supervised manner. When they train the system, weighted spectogram is used as an input and cross-entropy as a loss function.

    - _"Weighting STFT with proper perceptual weighting filters may overcome noise and pitch variability."_ Also, they applied some pre-processing like downsampling and hamming window.

- Use this pretrained NN to extract speaker embeddings which is time-dependent speaker charecteristic.

After that, system compare embeddings via cosine similarity. If difference is bigger than determined threshold, system say _this comes from different speaker_.

**Experiments**

- _"To evaluate our method and compare it with the state of the art, we use following publicly available datasets: AMI meeting corpus [39] (100 hours, 150 speakers), ISCI meeting corpus [40] (72 hours, 50 speakers), and YouTube (YT) speakers corpus [41] (550 hours, 998 speakers)._ Also, they release [open source dataset](http://github.com/cyrta/broadcast-news-videos-dataset) from broadcast material which comes from major new stations.

![alt text](https://docs.google.com/uc?id=1kbmjg2hszX0fuwd2gDjOk6wN_uTpVA97)

- They split the data into training and validation with the proportion of %70 and %30.
- Their baseline is state-of-art LIUM Speaker Diarization System which is baes on GMM classifier and uses 13 MFCC audio features as input. Also, they compare R-CNN with CNN via different features to understand effect of feature extraction.
- Their performance metric is _Diarization Error Rate(DER)_ for evaluation.

![alt text](https://docs.google.com/uc?id=1UlZMDQ9NbR4UkombtthzQeNIR2pM9Usc)


- _"The results of the evaluation can be seen in Tab. 2. Our proposed deep learning architecture based on recurrent convolutional neural network and applied to CQT-grams outperforms the other methods across all datasets with a large margin. Its improvement reaches over 30% with respect o the baseline LIUM speaker diarization method with default set of parameters."_



### 6) [_SPEAKER DIARIZATION WITH LSTM_](https://arxiv.org/abs/1710.10468)

_"In this paper, we build on the success of d-vector based speaker verification systems to develop a new d-vector based approach to speaker diarization. Specifically, we combine LSTM-based d-vector audio embeddings with recent work in non-parametric clustering to obtain a state-of-the-art speaker diarization system._

Their sistem is combination of 
- LSTM-based speaker verification model to extract speaker embeddings
- Non-parametric spectral clustering. They apply clustering algorithm to these embeddings in order to speaker diarization.

They obtain a state-of-art system for speaker diarization with this combination.

![alt text](https://docs.google.com/uc?id=1-kFDCGZ1itf4urm8JqOe9A_WoXwKn0qZ)

They have tried 4 different clustering algorithm in their paper. Two of them is belongs to online clustering (system label the segment when it is available, without seeing feature segments) and two of them is belongs to offline clustering (system label the segment when all segments are available). Offline clustering outperforms the online clustering.

- Naive online clustering, Links online clustering
- K-means offline clustering, **Spectral offline clustering**

Spectral offline clustering algorithm consists of the following steps: 

- Construct the affinity matrix. This matrix's elements represent the cosine similarity between segment embedding.

- Apply some refinement operations on the affinity matrix
    - Gaussian Blur to smooth the data. With this, reduce the effect of outliers.
    - Row-wise thresholding _(for each row, if elementssmaller than some threshold, set this element to 0)_
    - Symmetrization to restore matrix symmetry which is crucial for algorithm.
    - Diffusion to sharpen the matrix. Thus, we have more clear boundaries between speakers.
    - Row-wise max normalization to get rid of undesirable scale effects.

- Perform eigen decomposition.

##### For more info, please look the [paper](https://arxiv.org/abs/1710.10468)

![alt text](https://docs.google.com/uc?id=1N6KXrk_hdmS422YOU_-Wd-t3kVIpzYpk)


The writers discuss why we can not conventional clustering algorithms like K-mean. The problem comes from speech data's properties.
- *Non-Gaussian Distribution*: Speech data are often not-gaussian.
- *Cluster Imbalance*: For most of the recordings, mostly one speaker speaks. And if we use K-means, unfortunately, it can split this cluster into smaller cluster.
- *Hierarchical Structure*: The difference between one male and one female speaker is more than the difference between two male's clusters. This property, mostly, cause to K-means cluster all male's embeddings into one cluster and all female's embeddings into another cluster.

So that, they offer the novel *non-parametric spectral clustering* to solve these problems. 

**Experiment**

- VAD is used.
-- They use [_pyannote.metrics_](https://github.com/pyannote/pyannote-audio) library for evaluation.
- Fine tune parameters for each dataset.
- For CALLHOME dataset, they tolerate errors less than 250 ms in locating segment boundaries.
- Exclude overlapped speech.
- In general, they observed that d-vector based systems outperform i-vector based systems.
- They compare their result with state-of-art algorithms on CALLHOME dataset.

![alt text](https://docs.google.com/uc?id=1X1-cSylanhsYDKce1ThDj69kX2h_LiHt)


[_Poster of the paper_](https://sigport.org/sites/default/files/docs/icassp2018_diarization_poster.pdf)

### 7) [_Deep Speaker: an End-to-End Neural Speaker Embedding System_](https://arxiv.org/abs/1705.02304v1)

_"We present Deep Speaker, a neural speaker embedding system that maps utterances to a hypersphere where speaker similarity is measured by cosine similarity. The embeddings generated by Deep Speaker can be used for many tasks, including speaker identification, verification, and clustering."_

- They use the _ResCNN_ and _GRU_ to extract acoustic features.
- Mean pool to produce utterance level speaker embeddings.
- Train using _triplet loss_ based on _cosine similarity_.

Note: They use pre-training with a softmax layer and cross entropy over a fixed list of speaker. Thus, get better generalization and small loss.

**Architecture**

![alt text](https://docs.google.com/uc?id=1_8fP6YkrVX7FqhhmvEWxquQd1jbcX2HU)

- Firstly, preprocess the input, convert to 64-dimensional Fbank coeefficients. After that, normalize to zero mean and unit variance.
- Use feedforward-Dnn to extact features via ResNet or GRU.

![alt text](https://docs.google.com/uc?id=1KgDVoG5Cv5Qe5vs1Ok5T2OTEFqufzeR9)

- _Average sentence layer_ converts frame level input to an utterance-level speaker representation.
- _Affine_ and _Length Normalization_ layers map to speaker embeddings.
- After that, to train, they use the _triplet loss_. _"We seek to make updates such that the cosine similarity between the anchor and the positive exam- ple is larger than the cosine similarity between the anchor and the negative example"_ Thus, they can avoid suboptimal local minima.

![alt text](https://docs.google.com/uc?id=1v_VXnbPx6CsCJm6Q_jpKsdHqhXmtvAwr)

##### _For more info, please look the [paper](https://arxiv.org/abs/1705.02304v1)._

**Experiment**

- Outperforms a [DNN-based i-vector baseline](https://ieeexplore.ieee.org/document/6853887/). Both methods use VAD processing.
- They evaluate their method on three different dataset for speaker recognition task (both text-independent and text-dependent) in both Mandarin and English. 
- _"Speaker verification and identification trials were constructed by randomly picking one anchor positive sample (AP) and 99 anchor negative samples (AN) for each anchor utterance. Then, we com- puted the cosine similarity between the anchor sample and each of the non-anchor samples. EER and ACC are used for speaker verification and identification, respectively."_

_Text-Independent Results_

![alt text](https://docs.google.com/uc?id=1eijHzQmLIzD6n9WqmL9z1TH_krlexwjC)


_"In this paper we present a novel end-to-end speaker embedding scheme, called Deep Speaker. The proposed system directly learns a mapping from speaker utterances to a hypersphere where cosine similarities directly correspond to a measure of speaker similar- ity. We experiment with two different neural network architectures (ResCNN and GRU) to extract the frame-level acoustic features. A triplet loss layer based on cosine similarities is proposed for metric learning, along with a batch-global negative selection across GPUs. Softmax pre-training is used for achieving better performance."_

### 8) [_Unspeech: Unsupervised Speech Context Embeddings_](https://arxiv.org/abs/1804.06775v1)

Their method is based on _unsupervised_ learning. They train the system on up to 9500 hours English speech data with negative sampling method. They use _Siamese Convolutional Neural Network_ architecture to train _Unspeech_ embeddings. Their system is based on TDNN _(Time-Delayed Neural Network)_-HMM acoustic model to cluster.

Their idea comes from negative sampling in word2vec. 

##### Check [this blogpost](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) to understand negative sampling.

- System take the current segment as _target_ and target's left and right segments as _true context_. In addition to that, randomly sample four _negative context_.   

![alt text](https://docs.google.com/uc?id=1_aFaOp5J42oCErnNKYwQN6VNzyBXx25o)

- System use the _VGG16A_ network to convert these segments into embeddings.

![alt text](https://docs.google.com/uc?id=1WDST1HnAqGSM5W4YF07hhrFD8yL2H1iu)

- Train the system as binary classification task via logistic loss. 

**Experiment**

For same/different speaker experiment.

![alt text](https://docs.google.com/uc?id=1AdUzI8T3W8uy-HAZberEYDcoODe5sBhg)


[Preview of the paper from the writer](http://unspeech.net/preview/)

### 9) [_VoxCeleb2: Deep Speaker Recognition_](https://arxiv.org/abs/1806.05622)

_"In this paper, we present a deep CNN based neural speaker embedding system, named VGGVox, trained to map voice spectrograms to a compact Euclidean space where distances directly correspond to a measure of speaker similarity."_

This paper is related to _speaker verification_, however, their method and dataset can be useful to detect speaker change points. 

Their deep learning architecture consists of:
- Deep CNN _trunk_ architecture to extract features. 
    - *VGG_M and ResNet are their trunk architecture for this work. These works very well for image classification task. They just modified some part of these to make suitable for speech case.*
- Pooling layer to aggregate feature to provide a single embedding.
- Pairwise Loss 

They train *VGGVox* ,which is their neural embedding system, on short-term magnitude spectograms (a hamming window of width 25ms and step 10ms, without other pre-processing) in order to learn speaker discriminative embeddings via 2-step.

- Pre-training for identification using a softmax loss. With this pre-training, they can initialize their system weights.
- Fine-Tuning with the contrastive loss.

[Their dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2) both include audio and video.

![alt text](https://docs.google.com/uc?id=1L1G11lDbNAW4W20MTDSwi_qgEUW4sPwz)

**Experiment**

- They train the system on the _VoxCeleb2_ and test on the _VoxCeleb1_. They use _Equal Error Rate(EER)_ and their _cost function_ for evaluation. 
- _"During training, we randomly sample 3-second segments from each utterance._

![alt text](https://docs.google.com/uc?id=1_8wu8sYQBJVRK3u5877tEv4kXYRF_KFY)

_"In this paper, we have introduced new architectures and training strategies for the task of speaker verification, and demonstrated state-of-the-art performance on the VoxCeleb1 dataset. Our learnt identity embeddings are compact (512D) and hence easy to store and useful for other tasks such as diarisation and retrieval."_

### 10) [_TEXT-INDEPENDENT SPEAKER VERIFICATION USING 3D CONVOLUTIONAL NEURAL NETWORKS_](https://arxiv.org/abs/1705.09422)

This paper is about _speaker verification_, however, it can give some idea about how we can use 3D CNN to create speaker models to represent different speakers.

##### This project is open source. [Check it.](https://github.com/astorfi/3D-convolutional-speaker-recognition)

This work's novelty comes from usage of **3D-CNN** to capture speaker variations and extract the spatial and temporal information. _"The main idea is to use a DNN architecture as a speaker feature extractor operating at frame and utterance-level for speaker classification."_ 

Also they propose one shot learning to capture speaker utterances from the same speaker, instead of average the all d-vectors of the utterances of the targeted speaker. _"Our proposed method is, in essence, a one-shot representation method for which the background speaker model is created simultaneously with learning speaker characteristics._"

They compare their method with Locally-Connected Network(LCN) as a [baseline.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44681.pdf)
- This network uses locally-connected layers to extract low level features, fully-connected layers to extract high level features. 
- Loss function is cross-entropy for training. 
- During the evaluation phase, cosine similarity is used.

According to writers, this baseline method is not suitable to extract enough context of speaker related information, also baseline method is affected by non-speaker related information. To tackle these issues, they propose new model. Let's look their proposed architecture.
- 3D CNN architecture is suitable to capture both spatial and temporal information.
- Their input in the utterance level. 

![alt text](https://docs.google.com/uc?id=1Er3VN8kP9n27YP2iUgHmzoq15eLD1-6O)

- _"Our proposed method is to stack the feature maps for several different utterances spoken by the same speaker when used as the input to the CNN. So, instead of utilizing single utterance (in the development phase) and building speaker model based on the averaged representative features of different utterances from the same speaker (d-vector system)"_
- They apply pooling operation just for the frequency domain to keep useful information which is in the time domain.

**Experiment**

- They evaluate the model using the ROC (receiver operating characteristics) and PR (precision and recall) curves.

##### For more info, check the [paper](https://arxiv.org/abs/1705.09422)

- They use [WVU-Multimodal 2013 Dataset](https://github.com/astorfi/3D-convolutional-speaker-recognition/tree/master/data). _"The audio part of WVU-Multimodal dataset consists of up to 4 sessions of interviews for each of the 1083 different speakers."_

- They use modified MFCC as the data representation. MFCC has a drawback about its non-local characteristic because of last DCT operation for generating MFCC. Non-local input is not suitable for Convolutional NN, so that, they just discard the last DCT operation. Thus, they produce Mel-frequency energy coefficients (MFEC). In addition to that, window size is 20ms with 10ms stride.

- Their model outperforms the end-to-end training fashion.

![alt text](https://docs.google.com/uc?id=1qsY3IApwEK0qGdIgH2iYNO9dzwdjwt_V)

### 11) [_Deep Learning Approaches for Online Speaker Diarization_](http://web.stanford.edu/class/cs224s/reports/Chaitanya_Asawa.pdf)

Recently, there has been more work applying deep learning to speaker diarization problem.

- Learn speaker embeddings and use these embeddings to classify.
- Represent speaker identity using i-vectors.
- Bi-LSTM RNN 

In this paper, researchers have tried various strategies to tackle this problem.

- Speaker embeddings using triplet loss. (_inspired by Facenet_) Their model attempts to train a LSTM that can effectively encode embeddings for speech segments using the triplet loss.

    - _"In training, we take segments of audio from different speakers, and construct a triple that consists of (an anchor, a positive example, a negative example) where the anchor and positive example both come from the same speaker but the negative example is from a different speaker. We then want to generate embeddings such that the embedding for the anchor is closer to the positive example embedding by some margin greater than the distance from the embedding for the anchor to the negative example embedding."_

    - _"Then, when performing online diarization, we will run windows of speech through the LSTM to create an embedding for this window. If the produced vector is within some distance (using the L2 distance and a tuned threshold) of the stored current speaker vector, we deem that it is the same speaker. Otherwise, we detect that the speaker has changed, and compare the vector with the stored vector for each of the past speakers."_

![alt text](https://docs.google.com/uc?id=1OyIY9tihqcoulIvD4Dvg0l_TqPCV3Vv8)
 

Their proposed system can not capture some speaker changes which are short segments. Let's look their result for speaker change detection.

- _"Even in this task, we found that our models had difficulty capturing speaker change. As Figure 5 indicates, speakers are mostly speaking for few seconds each time they speak in a conversation – for example, we can imagine a lot of back-and-forth consisting of short segments: “(sentence)” “yeah” “(sentence)” “sure.” As a human listener, however, often these short snippets are looked over. This makes the problem of speaker detection very challenging because the model needs to rapidly identify that the speaker has changed and must also do this often."_


### 12) [_Blind Speaker Clustering Using Phonetic and Spectral Features in Simulated and Realistic Police Interviews_](http://oxfordwaveresearch.com/papers/IAFPA-2012-BlindClusteringAlexanderForthPresentation.pdf)

This paper is related to product of Oxford Wave Research called as _Cleaver_. They focus on the pitch tracking. According to them, if there is any significant discontunies either in time or frequency, is used to define a candidate transition between spekaers and cluster. Let's look their proposed method step by step.

- Take original speech and extract the pitch track with autocorrelation based pitch tracker.
- Perform the clustering which is based on pitch track continuities.
- Select the most similar(divergent) cluster 
- Make agglomerative clustering to improve the result for speaker clustering.
