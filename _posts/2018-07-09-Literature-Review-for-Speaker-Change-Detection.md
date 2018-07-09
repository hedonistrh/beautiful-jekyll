# Literature Review for Speaker Change Detection

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

### 3) [_TRISTOUNET: TRIPLET LOSS FOR SPEAKER TURN EMBEDDING_](https://arxiv.org/abs/1609.04301)

_"TristouNet is a neural network architecture based on Long Short-Term Memory recurrent networks, meant to project speech sequences into a fixed-dimensional euclidean space. Thanks to the triplet loss paradigm used for training, the resulting sequence embeddings can be compared directly with the euclidean distance, for speaker comparison pur- poses."_

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

- To solve class imbalance, the number of positive labels is increased artificially by labelling as positive every frame in the direct neighborhood of the manually annotated change point.

- Long audio sequences are split into short fixed-length overlapping sequences.

### 5) [_Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings_](https://arxiv.org/abs/1708.02840)

They are trying to solve speaker diarization problem via 2-step approach.

- To classify speaker, train a NN in a supervised manner. When they train the system, weighted spectogram is used as an input and cross-entropy as a loss function.
    - _"Weighting STFT with proper perceptual weighting filters may overcome noise and pitch variability."_ Also, they applied some pre-processing like downsampling and hamming window.

- Use this pretrained NN to extract speaker embeddings which is time-dependent speaker charecteristic.

After that, system compare embeddings via cosine similarity. If difference is bigger than determined threshold, system say _this comes from different speaker_.


### 6) [_Deep Learning Approaches for Online Speaker Diarization_](http://web.stanford.edu/class/cs224s/reports/Chaitanya_Asawa.pdf)

Recently, there has been more work applying deep learning to speaker diarization problem.
    - Learn speaker embeddings and use these embeddings to classify.
    - Represent speaker identity using i-vectors.
    - BiLSTM RNN 

In this paper, researchers have tried various strategies to tackle this problem.

- Speaker embeddings using triplet loss. (_inspired by Facenet_) Their model attempts to train a LSTM that can effectively encode embeddings for speech segments using the triplet loss.

    - _"In training, we take segments of audio from different speakers, and construct a triple that consists of (an anchor, a positive example, a negative example) where the anchor and positive example both come from the same speaker but the negative example is from a different speaker. We then want to generate embeddings such that the embedding for the anchor is closer to the positive example embedding by some margin greater than the distance from the embedding for the anchor to the negative example embedding."_

    - "_Then, when performing online diarization, we
    will run windows of speech through the LSTM to create an embedding for this window. If the produced vector is within some distance (using the L2 distance and a tuned threshold) of the stored current speaker vector, we deem that it is the same speaker. Otherwise, we detect that the speaker has changed, and compare the vector with the stored vector for each of the past speakers."_

![alt text](https://docs.google.com/uc?id=1OyIY9tihqcoulIvD4Dvg0l_TqPCV3Vv8)
 

Their proposed system can not capture some speaker changes which are short segments. Let's look their result for speaker change detection.

- _"Even in this task, we found that our models had difficulty capturing speaker change. As Figure 5 indicates, speakers are mostly speaking for few seconds each time they speak in a conversation – for example, we can imagine a lot of back-and-forth consisting of short segments: “(sentence)” “yeah” “(sentence)” “sure.” As a human listener, however, often these short snippets are looked over. This makes the problem of speaker detection very challenging because the model needs to rapidly identify that the speaker has changed and must also do this often."_


### 7) [_Blind Speaker Clustering Using Phonetic and Spectral Features in Simulated and Realistic Police Interviews_](http://oxfordwaveresearch.com/papers/IAFPA-2012-BlindClusteringAlexanderForthPresentation.pdf)

This paper is related to product of Oxford Wave Research called as _Cleaver_. They focus on the pitch tracking. According to them, if there is any significant discontunies either in time or frequency, is used to define a candidate transition between spekaers and cluster. Let's look their proposed method step by step.

- Take original speech and extract the pitch track with autocorrelation based pitch tracker.
- Perform the clustering which is based on pitch track continuities.
- Select the most similar(divergent) cluster 
- Make agglomerative clustering to improve the result for speaker clustering.
