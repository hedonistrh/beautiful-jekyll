# Literature Review for Speaker Change Detection

**Draft version. So that, there can be many typos and unreferenced quote. Also, please reach me, if you want to add different paper. Feel free to send e-mail to me.**

> UPDATE(17 October 2018): After the conversation with [Quan Wang](https://wangquan.me), I am trying to keep my blogpost up-to-date. I am very grateful because of his help and effort. I have added _FULLY SUPERVISED SPEAKER DIARIZATION_. ~~I will add Herve Bredin's new paper as soon as possible.~~


> UPDATE(29 October 2018): I have added _Neural speech turn segmentation and affinity propagation for speaker diarization_. 

> UPDATE(12 November 2018): Quan Wang's and his team publish the [source code](https://github.com/google/uis-rnn) of the [FULLY SUPERVISED SPEAKER DIARIZATION](https://arxiv.org/pdf/1810.04719.pdf)

> UPDATE(21 February 2019): Quan Wang release [the lecture](https://www.youtube.com/watch?v=pGkqwRPzx9U&list=PL3ik-FubnkJhHZxlT8wLwNed2D-LJ2PsB) about UIS-RNN. I highly recommend. 

Speaker diarization is the task of determining “who spoke when” in an audio stream that usually contains an unknown amount of speech from an unknown number of speakers. Speaker change detection is an important part of speaker diarization systems. It aims at finding the boundaries between speech turns of two different speakers.

![alt text](https://docs.google.com/uc?id=1ESCzgM-T9RRa-i7ozQFBR8ihUzO-OAPC)

##### **This slide is belongs to [this video](https://www.youtube.com/watch?v=pjxGPZQeeO4). I highly recommend it. :)**


Before papers, I just want to share some useful datasets.

- [Mozilla Common Voice](https://voice.mozilla.org/en/data)
- [Open SLR](https://www.openslr.org/resources.php) 
- [VoxCeleb2 from Oxford](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/)
- [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1)
- [TED-LIUM](http://www.openslr.org/7/)
- [TED-LIUM 3](https://www.openslr.org/51/)
- [LibriSpeech ASR corpus](https://www.openslr.org/12)
- [AMI-CORPUS](http://groups.inf.ed.ac.uk/ami/corpus/)
- [ISCI-CORPUS](http://groups.inf.ed.ac.uk/ami/icsi/)
- [CALLHOME](https://catalog.ldc.upenn.edu/ldc97s42)
- [WVU-Multimodal 2013 Dataset](https://biic.wvu.edu/data-sets/multimodal-dataset)


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

##### This project is [open source](https://github.com/pyannote/pyannote-audio).

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

### 5) [_Neural speech turn segmentation and affinity propagation for speaker diarization_](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1750.pdf)

They divide speaker diarization system to 4 sub-tasks:
- Speech Activity Detection (SAD)
- Speaker Change Detection (SCD)
- Speech Turn Clustering 
- Re-segmentation

Herve Bredin's previous [paper](https://pdfs.semanticscholar.org/edff/b62b32ffcc2b5cc846e26375cb300fac9ecc.pdf) explain the how they solve the Speech Activity Detection(SAD) and Speaker change Detection(SCD) via recurrent neural network, however, they used traditional methods to solve other 2 sub-taks at that paper. With these paper, they develop new approach to solve speaker diarization problem jointly.

![alt text](https://docs.google.com/uc?id=1nU8baMghp_nhas0F1XmXXPh2VTk5DcOW)


- Use LSTM for re-segmentation
- Use _Affinity propagation_ for speech turn clustering

We can list the contribution of this paper as:
- Adapt LSTM-based SAD and SCD with unsupervised resegmentation. _Previously, a GMM is trained for each cluster(speech segments which include same speaker) re-segment these with Viterbi decoding._

- Use affinitity propagation clustering on top of neural speaker embeddings. (_In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables. Neural network embeddings are useful because they can reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space._ [Source](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526))

- Joinly optimize whole steps which are LSTM-based SAD, LSTM-based SCD, LSTM-based speaker embeddings and LSTM-based re-segmentation (Just speech turn clustering is not based on RNN)

Now, let's deep dive into these contributions.

**Sequence Labeling based on LSTM**

At the previous paper, they used sequence labeling based on LSTM for speaker change detection and speech activity detection. With these modules, DNN create initial segmentation. (For more info, please check previous summary)

At that paper, they used same method (LSTM based) for re-segmentation. _Previously, re-segmentation is usually solved by GMMs and Viterbi decoding._ At test time, using the output of the clustering step (initial segmentation) as its unique training file, the neural network is trained for a tunable number of epochs _E_ and applied on the very same test file it has been trained on. After that, resulting sequence of K-dimensional score has been post-processed to determine new speech segments.

Drawback of this resegmentation is that increase false alarm.

**Clustering**

Speech turn clustering is solved by combination of he neural embeddings and affinitity propogation.

At the neural embedding stage, we are trying to embed speech sequences into a D-dimensional space. When we embed whole sequences into this space, we expect that if two sequences comes from same speaker, they will be closer in this space. _(Their angular distance will be small)_ To get embed of one segment, we need to process variable length segment, however, we should have fixed-length embeddings. To solve this problem, 
- Slide a fixed length window
- Embed each of these subsequences
- Sum these embedding

![alt text](https://docs.google.com/uc?id=1rbZLw0aYh0GsNEprj6pEIVaHcVWz6Bcx)

_The goal of SAD and SCD is to produce pure speaker segments containing a single speaker. The clustering stage is then responsible for grouping these segments based on speaker identities._

Herve Bredin and his team choose affinity propagation (AP) algorithm for clustering. Ap does not require a prior choice of the number of clusters. These means that, we do not have to specify how many speakers are there in the whole speech. All segments are potential for cluster centers. (These centers represent different speakers) When algorithm decide to examplers (cluster centers), it uses negative angular distance between embeddings to understand similarity. _(I do not want to give whole mathematics behind this algorithm. Please check [this wonderful blogpost](https://www.ritchievink.com/blog/2018/05/18/algorithm-breakdown-affinity-propagation/))_


**Joint Optimization**

Mostly, speaker diarization modules are tuned with empiricially(trial-and-error) Also, whole modules are tuned independently. Researcher use Tree-structured Parzen Estimator for hyper-parameter optimization. This method is available in [hyperopt](https://github.com/hyperopt/hyperopt).

**Experiments**

_This project is [open-source](github.com/yinruiqing/diarization_with_neural_approach). So, you can reproduce results. Herve Bredin and his team deserves Kudos. :)_

For feature extraction, they use Yaafe toolkit and use 19 MFCC, their first and second derivatives and the first and second derivatives of the energy. It means that, input is 59 dimensional.

For sequence labeling, SAD, SCD and re-segmentation modules share a similar network architecture.

![alt text](https://docs.google.com/uc?id=12u6UpGLv14Pd1_adbuQP28RCuFJ_tYKn)

For dataset and evaluation metric, they use French TV broadcast. 

![alt text](https://docs.google.com/uc?id=13S3JTC1aL_QtGtmVSFC5VpiMAoaJZA_Q)

They compare their results with two alternative approach
- Variant of the proposed approach, just they use standard hierarchical agglomerative clustering instead of affinity propogation
- S4D system which is developed by LIUM. This method use following approach:
_"Segmentation based on Gaussian divergence first generates (short) pure segments. Adjacent segments from the same speaker are then fused based on the Bayesian Information Criterion (BIC), leading to (longer) speech turns. Hierachical clustering based on Cross-Likelihood Ratio then groups them into pure clusters, further grouped into larger clusters using another i-vector-based clustering."_

![alt text](https://docs.google.com/uc?id=1U9TAr17NoThCs-ZlMSVCi9ChbFbA1KVo)

As we know from re-segmentation step, we need to determine _E_ to get best score. When we look at this figure, we can see that, we need same number of epochs for development and test set. This means that, LSTM-based re-segmentation is stable.

![alt text](https://docs.google.com/uc?id=1DB2Flihupva3rxihPfZPVS4qwR791C8P)



**Results and Conclusion**

- This pipeline is big step to reach integrated end-to-end neural approach to speaker diarization. Because, researcher show that initial segmentation and re-segmentation can be formulated via LSTM based sequence labeling.

- Affinity propagation outperforms the standart agglomerative clustering with complete-link.

**Future Direction**

- _"However, in re-segmentation step, finding the best epoch E relies on a development set. We plan to investigate a way to automatically select the best epoch for each file."_
- _"In addition, though neural networks can be used to embed and compare pairs of speech segments, it remains unclear how to do also cluster them in a differentiable manner."_

### 6) [_Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings_](https://arxiv.org/abs/1708.02840)

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



### 7) [_SPEAKER DIARIZATION WITH LSTM_](https://arxiv.org/abs/1710.10468)

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

Also, I highly recommend the [ICASSP Lecture](https://www.youtube.com/watch?v=AkCPHw2m6bY&t=18s) which is given by Quan Wang who is the writer of this excellent paper.

[![IMAGE ALT TEXT](https://img.youtube.com/vi/AkCPHw2m6bY/0.jpg)](http://www.youtube.com/watch?v=AkCPHw2m6bY "ICASSP 2018 Lecture Generalized End-to-End Loss for Speaker Verification")

Also, I can give brief information about the lecture. Some of them is not directly related with _speaker change detection_. However, it can gives excellent insight how to handle with problems.

- At Google, they use _2-stage speaker recognition: Enroll and verify_. Before the verification, user enrolls her voice with speak "OK Google" and "Hey Google". After that, they store the _averaged_ embedding vector.

- **Generalized end-to-end loss**: For the verification, they create the embedding from input via LSTM. After that, they compare the embeddings with cosine similarity. If similarity is bigger than threshold, system verify the user. To extract speaker embedding, we need define _loss function_. 
    - Most paper use Triplet Loss. It is very simple and can correctly models the embedding space, however, can not simulate _runtime behavior_. This means that it can not model averaging process. So that, it is not end-to-end.
    - In 2016, writers propose _tuple end-to-end loss_. It can model the averaging process. However, most tuples are very easy to train. So that, it is not very efficient.

    ![alt text](https://docs.google.com/uc?id=1kqzzQZ9uuhoxJ7ITxlDuW8qp8RbV6z5o)

    - To tackle with this problem, they propose _generalized end-to-end loss_. To train with this loss, they construct a similarity matrix for each batch. Also, in the video, you can see effiency comparision between TE2E and GE2E.

    ![alt text](https://docs.google.com/uc?id=1Q1_6Su8NuUKtr5IdYDN-DMz3xsBTA9ha)

- **Single Speaker Recognition Model For Multi-Keyword**: Their dataset have 150M "OK Google" utterances and 1.2M "Hey Google" utterances. To tackle with this class imbalance, they propose _Multi-Reader_. This combines the loss from batches of different data sources. It is like _regularization_.

![alt text](https://docs.google.com/uc?id=1omTp5yrYUc-d2ancOaaz_jERUfjXLybZ)

- **Text Independent Verification**: The challenge is that length of utterance can vary. Naive solution is full sequence training, however, it can be very slow. They propose the _sliding window inference_. When train the system, they use the batch which include same length.

![alt text](https://docs.google.com/uc?id=1kPzEWPF942O8vIzlvVD9ViN2ANMbU5yq)

##### Please check the [video](https://www.youtube.com/watch?v=AkCPHw2m6bY&t=18s) and [paper](https://arxiv.org/pdf/1710.10467.pdf) for results. Unfortunately, I can not cover all of them in this blog-post. 

##### **For the ICASSP's presenation of the paper, you can check [this video](https://www.youtube.com/watch?v=pjxGPZQeeO4). I highly recommend it. :)**

### 8) [_FULLY SUPERVISED SPEAKER DIARIZATION_](https://arxiv.org/pdf/1810.04719.pdf)

##### This project is open-source. Please check the [source code.](https://github.com/google/uis-rnn)

This paper comes from the writer who is the writer of [previous paper](https://arxiv.org/abs/1710.10468). Previous paper use unsupervised method for clustering, however, this paper use supervised method. So that, their method is fully supervised.

They called this system as _unbounded interleaved-state recurrent neural networks (UIS-RNN)_. **They use same baseline to extract d-vector with [previous paper](https://arxiv.org/abs/1710.10468).** After the extraction, each individual speaker is modeled by parameter-sharing RNN, while the RNN states for different speakers interleave in the time domain. With this method, system decodes in an online fashion. Also, their method is naturally integrated with ddCRP. Thus, system can learn how many speakers are there in the record. 

UIS-RNN method is based on three facts:

- We can model each speaker as an instance of RNN _(These instances share same parameters)_
- We do not have any constraint to specify speaker number. System can learn-guess how many speakers are there.
- The states of different RNN instances corresponding to different speakers. These different speakers interleaved in the time domain. 

#### **Overview of Approach**

_I do not want to give all mathematical backgrond. I am trying to simplify it._

- For the sequence embbeding, we will use **X**. This represent d-vector of a segment.
- For the ground truth label, we will use **Y**. For instance, Y = (1, 2, 2, 3, 3) These numbers represent speaker id. 

UIS-RNN is a generative process. 

![alt text](https://docs.google.com/uc?id=15Eq7u7P5eMjS8VIMNvUK-R5hTw5DopFp)

At that formula, we do not have speaker change information. So that, we define new parameter **Z** to represent speaker change. Now, we have augmented represenation.

![alt text](https://docs.google.com/uc?id=1IKfKgi88l3EnM-0pm6a7aktTyAXEJuIR)

For instance corresponding Z for the Y = (1 ,1 ,2, 3, 2, 2) is Z = (0, 1, 1, 1, 0). Because, when you look the Y, you can see that for second, third and fourth transition, there is a speaker change. So that, we write 1 at corresponding locations of Z.

_Note that, we can directly determine Z from Y. However, we can not uniquely determine Y from Z. Because, we can not know which speaker will come when there is a speaker change._

We can factorize the augmented representation.

![alt text](https://docs.google.com/uc?id=1aYFEQna5O0EX0L7WKL_d37_diqxKDcGO)

Now we have
- Sequence Generation
- Speaker Assigment
- Speaker Change

__Speaker Change__

z<sub>t</sub> represent speaker change. As we know from probability, z<sub>t</sub> is between 0 and 1. 

This can be parameterized via any function. However, writers use constant value for simplicity. So that it becomes binary variables.

![alt text](https://docs.google.com/uc?id=1w9VKpbvCFMD4uyE6cXQTh1vyE-QlsbBH)

**Speaker Assigment Process**

For the speaker diarization, one of the main challenge is that determine total number of speakers. For this challenge, researchers use  _distance dependent Chinese restaurant process (ddCRP)_ which is a Bayesian non-parametric model. 

When z<sub>t</sub> is 1, we know that there is a speaker change. At that point, there are 2 option. It can back to previously appeared speaker or switch to a new speaker.

- The probability of switching back to a previously appeared speaker is proportional to the number of continuous speeches she/he has spoken. - There is also a chance to switch to a new speaker, with a probability proportional to a constant α. 

![alt text](https://docs.google.com/uc?id=1NM-LKpALz9VdgtZSDw26fmPnzMeQsFfE)

__Sequence Generation__

_"Our basic assumption is that, the observation sequence of speaker embeddings X is generated by distributions that are parameterized by the output of an RNN. This RNN has multiple instantiations, corresponding to different speakers, and they share the same set of RNN parameters θ."_

They use GRU as RNN architecture to memorize long-term. 

State of GRU corresping to speaker z<sub>t</sub>:
 > m<sub>t</sub> = f(m<sub>t</sub>/θ) _This is the output of the entire newtork._

Let t' be the last time we saw speaker<sub>t</sub> before t

> t' := max{0, s < t : y<sub>s</sub> = y<sub>t</sub>} 


> h<sub>t</sub> = GRU(x<sub>s'</sub> , h<sub>s'</sub> /θ)

__Summary of the Model__

![alt text](https://docs.google.com/uc?id=1CuqgEY2VQeR795r3UjjNo9amRa8OvZqb)

##### Researcher omit Z and λ for simplicity. 

- Current stage, y<sub>[6]</sub> = (1, 1, 2, 3, 2, 2)

- There are four options, it can continue with same speaker which is 2, it can back to existing speakers which are 1 and 3 or it can swtich to a new speaker which will be 4. This is based on previous label assigment y<sub>[6]</sub> and previous observation sequence x<sub>[6]</sub>

##### I will skip details of MLE and MAP for sake of simplicity. For the details, please check the excellent paper.

- **For Training**

    System will try to maximize MLE estimation. 

    ![alt text](https://docs.google.com/uc?id=1yGizh4XcYfINWrn8Ukc7GZ3Rue9khCmJ)

- **For Testing**

    System will decode and ideal goal is to find:

    ![alt text](https://docs.google.com/uc?id=1IrK446DfnOAxiKPPmIU1qkEUn9k5HaHB)

**Experiments and Results**

- **Speaker Recognition Model**

    They use three different model.

    - “d-vector V1”. This model is trained with 36M utterances from 18K US English speakers, which are all mobile phone data based on anonymized voice query logs
    - "d-vector V2". More training data has been added to V1.
    - “dvector V3” retrained by using variable-length windows, where the window size is drawn from a uniform distribution within [240ms, 1600ms] during training.

    Results for speaker verification task.

    ![alt text](https://docs.google.com/uc?id=1Y4TR3wawMg4Kvkl-x9BM7I20xrSs98GU)

- **UIS-RNN Setup**
    - One layer of 512 GRU cells with a tanh activation
    - Followed by two fully-connected layers each with 512 nodes and a ReLU activation. 
    - The two fully-connected layers

For the evaluation they use [pyannote.metrics](http://pyannote.github.io/pyannote-metrics) as evaluation metrics and NIST Speaker Recognition Evaluation as dataset.

![alt text](https://docs.google.com/uc?id=1QCtubDJ0PTe524DUKjhCmRBWh75ohqpK)

As we can see, when they use V3, their result significantly improved. Because, it uses variable-length windows. 

UIS-RNN can beat offline-clustering methods even produces speaker labels in an **online** fashion.

**Conclusions**

_"Since all components of this system can be learned in a supervised manner, it is preferred over unsupervised systems in scenarios where training data with high quality time-stamped speaker labels are available."_

### 8) [_Deep Speaker: an End-to-End Neural Speaker Embedding System_](https://arxiv.org/abs/1705.02304v1)

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

### 9) [_Unspeech: Unsupervised Speech Context Embeddings_](https://arxiv.org/abs/1804.06775v1)

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

### 10) [_VoxCeleb2: Deep Speaker Recognition_](https://arxiv.org/abs/1806.05622)

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

### 11) [_TEXT-INDEPENDENT SPEAKER VERIFICATION USING 3D CONVOLUTIONAL NEURAL NETWORKS_](https://arxiv.org/abs/1705.09422)

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

### 12) [_Deep Learning Approaches for Online Speaker Diarization_](http://web.stanford.edu/class/cs224s/reports/Chaitanya_Asawa.pdf)

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


### 13) [_Blind Speaker Clustering Using Phonetic and Spectral Features in Simulated and Realistic Police Interviews_](http://oxfordwaveresearch.com/papers/IAFPA-2012-BlindClusteringAlexanderForthPresentation.pdf)

This paper is related to product of Oxford Wave Research called as _Cleaver_. They focus on the pitch tracking. According to them, if there is any significant discontunies either in time or frequency, is used to define a candidate transition between spekaers and cluster. Let's look their proposed method step by step.

- Take original speech and extract the pitch track with autocorrelation based pitch tracker.
- Perform the clustering which is based on pitch track continuities.
- Select the most similar(divergent) cluster 
- Make agglomerative clustering to improve the result for speaker clustering.
