# ProcessTransformer: Predictive Business Process Monitoring with Transformer Network

Zaharah A. Bukhsh , Aaqib Saeed , and Remco M. Dijkman

Eindhoven University of Technology, Eindhoven, The Netherlands {z.bukhsh, a.saeed, r.m.dijkman}@tue.nl

Abstract. Predictive business process monitoring focuses on predicting future characteristics of a running process using event logs. The foresight into process execution promises great potentials for eficient operations, better resource management, and efective customer services. Deep learning-based approaches have been widely adopted in process mining to address the limitations of classical algorithms for solving multiple problems, especially the next event and remaining-time prediction tasks. Nevertheless, designing a deep neural architecture that performs competitively across various tasks is challenging as existing methods fail to capture long-range dependencies in the input sequences and perform poorly for lengthy process traces. In this paper, we propose <sup>ProcessTrans-</sup> <sup>former</sup>, an approach for learning high-level representations from event logs with an attention-based network. Our model incorporates long-range memory and relies on a self-attention mechanism to establish dependencies between a multitude of event sequences and corresponding outputs. We evaluate the applicability of our technique on nine real event logs. We demonstrate that the transformer-based model outperforms several baselines of prior techniques by obtaining on average above 80% accuracy for the task of predicting the next activity. Our method also perform competitively, compared to baselines, for the tasks of predicting event time and remaining time of a running case.

Keywords: Predictive process monitoring · transformer · attention · deep learning · activity prediction · remaining time prediction

## 1 Introduction

With the trend towards digital transformation and the availability of relatively cheaper storage solutions, the amount of process execution data (also referred to as event logs) are increasing at a tremendous scale. Process mining methods have been used to discover, monitor, and improve the business processes by analyzing the process logs. Instead of post-hoc analysis, organizations are actively investing in predictive analytics solutions to gain insights into their performance. Predictive business process monitoring (PBPM) has emerged as a crucial area of process mining, focusing on estimating future characteristics of a running business process. PBPM has several useful business applications, including efective resource management, improving operational eficiency, and avoiding deadlock by predicting the next possible activities, duration, and remaining time to completion.

In the last decade, deep neural networks are widely adopted for tasks related to business process monitoring. Evermann et al.,[10] introduced recurrent neural networks (RNNs) for predicting the process behavior at run-time. Following this, several prediction models based on RNNs and related variants, such as long-short term memory (LSTM) networks have been proposed for the tasks of next activity [33,21,35], sufix generation [18,4], outcome prediction [36], and process’s remaining time prediction [27,20]. Even though deep RNNs based PBPM methods have been firmly established for event sequence modeling, they sufer from considerable shortcomings, specifically for the next activity prediction task.

Firstly, several proposed methods [33,4,20,21,35] employ one-hot encoding to obtain the numeric representations of categorical events sequences. These integer representations disregard the intrinsic relationship among events and introduce unrealistic computational requirements due to an increase of data dimensionality [12]. Secondly, LSTM lacks the explicit modeling of long and short-range dependencies in the sense that their performance degrades in proportion to the length of events sequences [23]. It is specifically undesired for event logs due to interconnections introduced by control flows among activities. Lastly, the inherent sequential nature of LSTM and RNNs precludes parallelization, resulting in critically ineficient learning and inference.

The attention mechanism is proposed to address the problem of long-range dependencies for sequence modeling without regard to their distance in input and output sequence [2]. In particular, Vaswani et al. [37] introduced Transformer neural network architecture, a deep sequence model that employs selfattention to maintain coherence in long-range sequences. The Transformer-based encoder-decoder models have rapidly become a dominant architecture for neural machine translation and natural language understanding [40]. Specifically, Transformer architecture is also behind the compelling language models, such as GPT-3 (Generative Pretrained Transformer) and BERT (Bidirectional Encoder Representations from Transformers) that have revolutionized numerous language understanding tasks. An interested reader may refer to [2] for a comprehensive explanation about attention-based networks.

Despite showing remarkable performance in multiple sequence modeling problems, Transformers have not been explored in the realm of business process management. Thus, the core contribution of our work is the <sup>ProcessTrans-</sup> <sup>former</sup> model with an improved strategy for learning high-level generic representations directly from temporal sequential events with minimal preprocessing of the input. In contrast to fixed-size recurrent memory models, the selfattention mechanism enables access to any part of the previously generated events in a sequence. It allows the deep neural network to capture global dependencies between inputs and outputs for powerful general-purpose representation learning. Besides, <sup>ProcessTransformer</sup> can efectively diferentiate the most relevant features that afect the model prediction. We demonstrate the applicability of <sup>ProcessTransformer</sup> on nine real event logs. We show that our Transformer-based architecture outperforms several baselines of existing methods on predicting the next activity task with minimal data preprocessing. Similarly, <sup>ProcessTransformer</sup> also shows performance improvements for predicting the next event time and completion time of a running case.

The remainder of the paper is structured as follows. Section 2 provides the background and related work. Section 3 provides definitions of main concepts related to PBPM. Section 4 presents the proposed approach, followed by Section 5, which details the experimental setup and key results. Finally, Section 6 summarizes the findings, contributions and outlines the future work.

## 2 Background and Related work

Representation learning focuses on extracting discriminative features from raw unstructured data to efectively solve the prediction problem, ideally in an endto-end manner. The general-purpose representations learned from a plethora of raw, real-life event logs can be used to solve several business tasks of interest, including predictive process monitoring, improvement, and enhancement.

In this section, we introduce sequence modeling in the context of deep learning. We also provide a brief overview of literature studies related to PBPM.

## 2.1 Sequence modeling using deep neural architectures

Sequence modeling involves capturing high-level semantic relationships in a series of interdependent input values that can be useful for various tasks, such as text completion or predicting the next word in a sentence. Compared to standard independent and identically distributed datasets, elements in the sequence follow a certain order and are not independent of each other. Typical machine learning algorithms and standard feedforward networks fall short in sequence modeling due to their inability to handle the order and keep the memory of past seen samples [32]. Natural language processing, time-series analysis, and predictive process monitoring are the key areas for sequence modeling.

Recurrent neural networks (RNNs) introduced an internal memory state to retain the memory of past inputs. The hidden layer within a recurrent network receives input from the input layer and the previous hidden layer’s output at each timestep. However, RNNs cannot maintain the context information for long input sequences and sufer from gradient vanishing or exploding problems due to the recursive derivation of gradients during model training.

Long Short Term Memory networks (LSTMs) are special kinds of RNNs with multiple switch gates to avoid the gradient vanishing problem and to remember long-range input dependencies [13]. Even though LSTMs perform better than RNNs, they sufer from similar limitations as RNNs when an input sequence become excessively long. Additionally, LSTMs are computationally expensive to train due to long sequential gradient paths which makes it harder to parallelize them.

Transformer is introduced by Vaswani et al. [37] for neural machine translations. Essentially, the transformer replaced the recursive approach of previously introduced recurrent networks with the self-attention mechanism. It enables the transformer to reason over long-range dependencies and draw generic representations between input and output. Self-attention mechanism decides the importance of all tokens in a sequence with respect to the input token. In Figure 1, we introduce the self-attention mechanism with the help of an example. A model based on selfattention takes a trace having six events as input. Each event is represented with embedding vectors called a query, key, and values. To compute the attention representation vector w<sub>j</sub> of Ship product event, we must measure how it relates to the other events in a trace. For this, we take the dot product of the query vector of interest with the keys of all input

Fig. 1. An example of Selfattention mechanism to learn attention representation vector of Ship product event.

vectors, resulting in a vector of weights $a _ { j }$ for all the input tokens. The attention vector is then computed by taking the weighted sum of $a _ { j }$ with value vectors. In other words, every output is a weighted sum of every input. The attention representation for each input token can be computed in parallel since their operations are independent, thus eliminating the recurrence. A single event can be related to other events in a sequence in multiple ways, such as semantically, temporally, among others. Therefore, the self-attention is projected multiple times to capture all of these dependencies, forming the multi-head self-attention.

The transformer is mainly used for language modeling; however, it is a generic network that can be adopted for multiple sequence modeling tasks. In this paper, we propose the transformer model to address the PBPM tasks. Further details of ProcessTransformer are given in Section 4.

## 2.2 Predictive process monitoring

PBPM has emerged as a promising area of process mining, having a wide array of business applications. Initially, the research focus has been on examining the process outcome in terms of its duration and successful completion. Multiple process analytics techniques based on hidden Markov model [22], finite state machines [11], stochastic Petri nets [29], and annotated transition systems (using diverse abstractions) [1] have been proposed. Classical machine learning classification methods, such as random forest and support vector machines, are also adapted to predict the successful completion of a process using the history of event sequences and hand-crafted features [5,14].

With the automated features learning capabilities of deep neural networks, specifically LSTMs, deep learning methods have been adopted to extract useful representations from large-scale temporal sequential event logs to solve various tasks. Successful application of deep sequence modeling has been explored earlier by Evermann et al. [10] for predicting the next event using a shallow LSTM model and embedding technique for handling categorical variables. A similar LSTM model architecture with one-hot vector encoding was employed by Tax et al. [33] to predict the next activity with its associated timestamp, remaining process duration, and process sufix. Likewise, the same process monitoring problems were addressed by Camargo et al. [4] using the composition of LSTM to support both categorical and numeric features. Additional temporal features are proposed in [20] and [21] to improve the predictive capabilities of existing deep models. An extension of LSTM with attention mechanism is used in [38] for the process outcome prediction task.

Although LSTM has been a popular choice due to its sequence modeling characteristics, other architectural variants of deep neural networks and analytical techniques are also explored in few studies. Prominently, Khan et al., [15] introduced memory augmented neural networks as a recommendation tool for tackling complex process analytic problems. Pasquadibisceglie et al., [24] proposed a data engineering approach to transform events temporal data to spatial image-like structure in order to use the convolution neural networks (CNN). Similarly, Mauro et al., [6] adapted the inception architecture of CNN for sequential data to address the next activity prediction problem. Pauwels et al. [25] presented a Bayesian technique to predict the next event. Taymouri et al., [34] adapted generative adversarial nets (GANs) with Gumbel-Softmax distribution to use them for (categorical) sufix generation and remaining time prediction. Bohmer et al. [3] proposed combining local and global techniques using sequential prediction rules for the next event prediction. For further relevant work on PBPM, an interested reader may refer to [28] and [39] survey studies.

## 3 Preliminaries

In this section, we introduce concepts that will be used to define the problem and data preprocessing for predictive process modeling in the subsequent sections. We follow the standard notations provided in [33] and [28].

Definition 1 (Event) Let A be the set of activities, C the set of cases, T the time domain and $D _ { 1 } , . . , D _ { m }$ the set of related attributes where $m > 0$ . An event is a tuple $e = ( a , c , t , d _ { 1 } , \ldots , d _ { m } )$ , where $a \in A , c \in C , t \in T$ and $d _ { i } \in \{ D _ { i } \}$ with $i \in [ 1 , m ]$

Definition 2 (Trace, Events Log) Let $\pi _ { A } , \ \pi _ { C } .$ and π<sub>T</sub> be functions that map an event $e = ( a , c , t , d _ { 1 } , \ldots , d _ { m } )$ to an activity, as $\pi _ { A } ( e ) = a$ , to a unique case identifier, as $\pi _ { C } ( e ) = c$ and to a timestamp, as $\pi _ { T } ( e ) = t . \mathrm { ~ A ~ }$ trace is defined as a finite non-empty sequence of events $\sigma = \langle e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \rangle$ , such that $\forall e _ { i } , e _ { j } \in \sigma _ { \ i }$ , it must hold that: the events within a trace σ must have same case id, i.e. $\pi _ { C } ( e _ { i } ) = \pi _ { C } ( e _ { j } )$ and time should be non-decreasing, i.e. $\pi _ { T } ( e _ { j } ) \geq \pi _ { T } ( e _ { i } )$

for $j > i ,$ . We say that a trace $\sigma = \langle e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \rangle$ has length $n ,$ denoted $| \sigma |$ . An event log is collection of traces $L = \{ \sigma _ { 1 } , \sigma _ { 2 } , . . . , \sigma _ { l } \}$ . We say that a collection $L = \{ \sigma _ { 1 } , \sigma _ { 2 } , . . . , \sigma _ { l } \}$ has size $l ,$ denoted $| L |$

Definition 3 (Activity Prediction) Let σ be a trace $\langle e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \rangle$ and $k \in$ $[ 1 , n - 1 ]$ be a scalar positive number. The event prefix of length $k , h d ^ { k }$ can be defined as: $h d ^ { k } ( \sigma ) = \langle e _ { 1 } , e _ { 2 } , \ldots , e _ { k } \rangle$ . The activity prefix can be obtained by the application of mapping function $\pi _ { A }$ as $\pi _ { A } ( h d ^ { k } ( \sigma ) ) = \langle \pi _ { A } ( e _ { 1 } ) , \pi _ { A } ( e _ { 2 } ) , \ldots , \pi _ { A } ( e _ { k } ) \rangle$ Activity prediction is the definition of a function $\Theta _ { a }$ that takes event prefix $h d ^ { k } ( \sigma )$ where $k \in [ 1 , n - 1 ]$ , and predicts the next activity $e ^ { \prime } { \mathrm { . } }$ , i.e.:

$$
\Theta_ {a} (h d ^ {k} (\sigma)) = \pi_ {A} (e _ {k + 1} ^ {\prime})
$$

Definition 4 (Event Time Prediction) Let $\sigma = \langle e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \rangle$ be a trace of length $n .$ To extract the time-related features of the last event $e _ { n }$ of that trace we define the functions:

$$
\begin{array}{l} f v _ {t 1} (\sigma) = \left\{ \begin{array}{l l} 0 & \text {if | \sigma| = 1 ,} \\ \pi_ {T} (e _ {n}) - \pi_ {T} (e _ {n - 1}), & \text {otherwise.} \end{array} \right. \\ f v _ {t 2} (\sigma) = \left\{ \begin{array}{l l} 0 & \text {if | \sigma| \in [1,2],} \\ \pi_ {T} (e _ {n}) - \pi_ {T} (e _ {n - 2}), & \text {otherwise.} \end{array} \right. \\ f v _ {t 3} (\sigma) = \left\{ \begin{array}{l l} 0 & \text {if | \sigma| = 1 ,} \\ \pi_ {T} (e _ {n}) - \pi_ {T} (e _ {0}) & \text {otherwise.} \end{array} \right. \end{array}
$$

The $f v _ { t 1 }$ feature represents the time diference between the previous event and the current event of a trace. The $f v _ { t 2 }$ feature contains the time diference between current event time and time of an event before the previous event. Finally, $f v _ { t 3 }$ depicts the approximate time passed since the case has initiated. (Note that we say the approximate time, because, due to the fact that we only have the completion time of each event, we do not know the duration of the first event or the time the case waited for the first event to occur.) Event time prediction is the definition of a function $\Theta _ { t }$ that takes event prefix $h d ^ { k } ( \sigma )$ where $k \in [ 1 , n - 1 ]$ 2 and predicts the time moment at which the next activity will occur, i.e.:

$$
\Theta_ {t} (\sigma^ {\prime}, f v _ {t 1} (\sigma^ {\prime}), f v _ {t 2} (\sigma^ {\prime}), f v _ {t 3} (\sigma^ {\prime})) = \pi_ {T} (e _ {k + 1} ^ {\prime}), \mathrm{where} \sigma^ {\prime} = h d ^ {k} (\sigma)
$$

Definition 5 (Remaining Time Prediction) Let $\sigma = \langle e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \rangle$ . Remaining time prediction is the definition of a function $\Theta _ { r t }$ that takes event prefix $h d ^ { k } ( \sigma )$ where $k \in [ 1 , n - 1 ]$ , and predicts the remaining time of the case, i.e.:

$$
\Theta_ {r t} (\sigma^ {\prime}, f v _ {t 1} (\sigma^ {\prime}), f v _ {t 2} (\sigma^ {\prime}), f v _ {t 3} (\sigma^ {\prime})) = \pi_ {T} (e _ {n}) - \pi_ {T} (e _ {k}), \mathrm{where} \sigma^ {\prime} = h d ^ {k} (\sigma)
$$

Note that $f v$ functions are applied manually for feature extraction as a preprocessing step, whereas Θ functions are learned in an end-to-end manner <sub>with</sub> ProcessTransformer <sub>.</sub>

## 4 Process Transformer

Real-life event logs present temporally sequential data, which is complex, variable, has extensive dependencies due to multiple control flows. Recurrent neural networks, such as LSTM, struggle to reason over long-range sequences due to the limited size of a context vector, as noted in [23]. This paper addresses the problem of PBPM to predict the next activity, event time, and remaining time of a process under execution, i.e., using deep learning to learn the functions $\Theta _ { a } , \Theta _ { t } ,$ and $\Theta _ { r t }$ as they are defined in Definitions 3-5. To that end we propose the <sup>ProcessTransformer</sup> . Notably, while several deep learning-based process monitoring methods [33,4,24] exist, which learn a predictive model based on varying prefixes length of event sequences, we develop a deep neural network that considers possible prefixes altogether for training and inference.

Figure 2 provides a high-level overview of the model architecture. The <sup>Pro-</sup> <sup>cessTransformer</sup> has N attention blocks, which take positional encoded input sequence and pass learned representation to a pooling operation and then to a fully connected layer. We use single attention block for <sup>ProcessTransformer</sup> . The attention block comprises multi-headed attention layers, followed by feedforward layers having residual connections, dropout, and a normalization layer. In the following, we explain the most important building blocks of our model.

Fig. 2. Model Architecture of Process Transformer.

Trace Embedding and Positional Encoding: Starting from a trace, the network learns vector embedding for each event as shown in the respective block in Figure 2. It essentially maps the categorical input to the vector of continuous representations. The network learns vector embedding and projects them into a transformed space where similar events in a semantic sense are mapped closer to each other. The benefit of using embedding in this way is that it eliminates the problem of high-dimensionality encountered in a one-hot encoding scheme. For example, while a one-hot encoding would require a binary vector for each event depending on the size of vocabulary (unique event instances), the embedding layer learns representation of semantically similar events by mapping them individually to a vector space.

It is worth noting that the <sup>ProcessTransformer</sup> does not use recurrence as former approaches (e.g., in [33,4]). This property enables eficient training but at the cost of omitted positional information of events from traces. The default Transformer architecture proposed to add positional encoding along with an input embedding to inform the model about the relative positioning of each token in a sequence. We choose the 36 -dimensional vector encoding to represent the relative positioning of an event in a trace. The input embedding and position encoding have the same dimension to allow for their summation at the next step. The neural model learns to attend to positional encoding along with input embedding in an end-to-end manner during learning to solve a specific task.

Self-Attention (Scaled Dot-Product) Self-attention models learn to selectively attend to only important parts of a trace to compute the robust representation. It enables the Transformer to reason over long-range dependencies and draw generic representations between input and output. We illustrate the selfattention mechanism with the help of an example trace of order management process in Figure 3. It shows that the events such as receive order and check credit, obtain product and ship product are related and will have high relative attention scores with respect to each other. The events like update inventory, send invoice, and check credit are semantically diferent in the context of order management and may have low attention scores. This self-attention enables the model to pay attention to important event(s) of a trace in order to better solve the predictive monitoring tasks.

Fig. 3. Illustration of self-attention mechanism on a trace. The model learns attention scores of events for solving a particular predictive monitoring task.

The attention mechanism initially creates three vectors called query $q ,$ key k and value v for each input embedding i.e. $x _ { 1 }$ , as shown in Figure 2. The neural model learns the representations for these vectors. A self-attention function maps a query $q$ to a set of key-value pairs, denoted as k and $v ,$ to obtain a weighted sum of values called output, denoted by $\mathcal { Z } .$ Following [37], we adopt scaled dotproduct attention to compute $\mathcal { Z }$ as follows:

$$
\mathrm{Attention} (Q, K, V) = \mathcal {Z} = \mathrm{softmax} (\frac {Q K ^ {T}}{\sqrt {d _ {k}}}) V
$$

where $Q , K ,$ and V are matrices of $q , k ,$ and v vectors packed into respective matrices for eficient computations. The attention scores are scaled by dividing to $\sqrt { d _ { k } }$ i.e., dimension of key $k ,$ for stable gradient computations. Afterward, the softmax function is applied to normalize the scores and obtain the importance of each token. Finally, the softmax score is multiplied with the value V matrix to give the model capability to focus on which words to attend to in a sequence and eliminate less relevant tokens.

Multi-Head Self Attention A single event can be related to other events in a trace in multiple ways, such as semantically, temporally, among others. In order to introduce a model with diferent representation subspaces at diferent positions [37], the query, key, and values are linearly projected h times as shown in Figure 2. The scaled dot-product attention is performed in parallel for each of the projection, forming the multi-headed attention as follows:

$$
\begin{array}{c} \operatorname{MultiHead} (Q, K, V) = \operatorname{Concat} (h e a d _ {1}, \ldots , h e a d _ {h}) W ^ {O} \\ h e a d _ {i} = \operatorname{Attention} (Q W _ {i} ^ {Q}, K W _ {i} ^ {K}, V W _ {i} ^ {V}) \end{array}
$$

where the output of each head<sub>i</sub> is linearly concatenated and multiplied by weight matrix $W ^ { O }$ , that is learned by the deep neural network.

A dropout layer follows the output of the multi-head self-attention block to limit over-fitting, and a layer normalization is applied across the layer’s features. Afterward, a position-wise feed-forward layer followed by a dropout and layer normalization is applied. The attention blocks $\mathcal { N }$ make use of the residual connection, depicted by dashed lines in Figure 2, to enable gradients to skip non-linear functions, thus avoiding the vanishing gradient problem. The rest of the layers after the attention block $\mathcal { N }$ are mainly network design choices, which are covered in detail in the following section.

Network Design and Implementation Given an event log, we create a word dictionary to encode the event names numerically. The traces that are shorter than $L _ { | \sigma _ { n } | }$ (i.e., the maximum length of a trace in event logs) are padded with zero to have a fixed-length input. We transform each event in a trace using a learnable distributed embedding of 36 units and a positional encoding with the same dimension. The embedding outputs are then fed to a multi-headed attention block with $h \ = \ 4 \ ( h$ being the number of heads) in order to learn generalpurpose representations at diferent input positions. We apply a global maxpooling on the last layer of the attention block to aggregate features, followed by a dropout with a rate of 0.1. We also use dense layers with 32 and 128 hidden units having ReLU activation consecutively. A fully-connected layer with hidden units corresponding to the output dimension of a predictive task is used, with either softmax or linear activation functions (depending on the target task), to produce output.

We employ categorical cross-entropy loss for the next activity prediction task. Likewise, we use the log-cosh loss function for regression-based tasks, i.e., for the event time and remaining time prediction problem. Here, we concatenate the output of the attention block with scaled numeric temporal attributes. The rest of the architecture remained the same across tasks.

## 5 Evaluation

We evaluate the eficacy of the proposed <sup>ProcessTransformer</sup> on nine reallife event logs. We also provide comparisons to established benchmarks reported in [28]. This section introduces the experimental setup, including datasets and evaluation metrics, followed by input preprocessing and network design details. We conclude the section by providing evaluation results of three predictive process monitoring tasks namely next activity, next event time and remaining time prediction of a running case.

## 5.1 Experimental setup

Datasets The experiments were conducted using event logs publicly available at the 4TU Research Data repository<sup>1</sup>. Some of these datasets have been widely used to evaluate process monitoring tasks (e.g. [4,18,33,10]). Table 1 provides the descriptive statistics of considered event logs.

Table 1. Descriptive statistics of event logs used for evaluations. Time-related characteristics are reported in days.

<table><tr><td>Datasets</td><td>Cases</td><td>Events</td><td>Activities</td><td>Max case length</td><td>Avg. case length</td><td>Max case duration</td><td>Avg. case duration</td></tr><tr><td>Helpdesk [26]</td><td>4,580</td><td>21,348</td><td>14</td><td>15</td><td>4.66</td><td>60</td><td>40.69</td></tr><tr><td>BPIC12 [7]</td><td>13,087</td><td>262,200</td><td>24</td><td>175</td><td>20.03</td><td>13</td><td>8.01</td></tr><tr><td>BPIC12w [7]</td><td>9,658</td><td>170,107</td><td>7</td><td>156</td><td>17.61</td><td>132</td><td>10.5</td></tr><tr><td>BPIC12cw [7]</td><td>9,658</td><td>72,413</td><td>6</td><td>74</td><td>7.497</td><td>82</td><td>10.46</td></tr><tr><td>BPIC13 [31]</td><td>7,554</td><td>65,533</td><td>13</td><td>123</td><td>8.6754</td><td>768</td><td>11.948</td></tr><tr><td>BPIC20d [8]</td><td>10,500</td><td>56,437</td><td>17</td><td>24</td><td>5.37</td><td>47</td><td>11.16</td></tr><tr><td>BPIC20i [9]</td><td>6,449</td><td>72,151</td><td>34</td><td>27</td><td>11,187</td><td>737</td><td>84.15</td></tr><tr><td>Hospital [19]</td><td>100,000</td><td>451,359</td><td>18</td><td>217</td><td>4.51</td><td>1034</td><td>127.24</td></tr><tr><td>Traffic fines [17]</td><td>150,370</td><td>561,470</td><td>11</td><td>20</td><td>3.73</td><td>4373</td><td>342.67</td></tr></table>

Evaluation metrics For the next activity prediction task, accuracy is a commonly used metric as reported in several studies [33,15,24,10]. Accuracy is essentially computed by taking a fraction of correctly predicted samples to the total number of samples. However, in the case of an imbalanced dataset, accuracy as performance metric can be misleading. Therefore, we report weighted accuracy and weighted F-score in the paper. The weighted aspect considers the data imbalance of the target class and assigns the weights to data samples accordingly. The F-score is a combination of precision (or positive predictive value) and recall (sensitivity) measures [30]. The precision determines the exactness of the model, whereas the recall provides a measure of the model’s completeness. F-score is calculated as follows:

$$
\mathrm{F-score} = 2 \times \frac {\mathrm{precision} \times \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$

For next event time and remaining time prediction having continuous target output, we compute mean absolute error (MAE) as follows:

$$
\mathrm{MAE} = \frac {\sum_ {i = 0} ^ {n} | y _ {i} - \hat {y _ {i}} |}{n}
$$

where $y _ { i }$ is the predicted value from the model, $\hat { y } _ { i }$ is the desired output, and n is the total number of samples in the test set.

## 5.2 Data preprocessing and training setup

The event logs are first chronologically ordered to simulate the reality in which a model uses past traces to monitor the future performance of running traces. Each dataset is split into 80% and 20% for training and testing sets, respectively, while preserving its temporal order. Additionally, 20% of the data from the training split is used for validation and hyper-parameter tuning during the learning phase.

For training and evaluation of <sup>ProcessTransformer</sup> network, following key points are important:

– We use the raw event logs data with minimal preprocessing. This means we did not selectively filter out events with specific k-prefixes, i.e. $h d ^ { k } ( \sigma )$ , unlike noted in [33,34]. The use of all prefixes enables the model to perform process monitoring tasks for extremely small (e.g., single event) to very lengthy running cases.

The model is trained for 100 epochs with an ADAM optimizer [16] and a learning rate of $1 0 ^ { - 2 }$ for all the considered tasks. Furthermore, we explore the impact of the batch size and the number of attention heads. We report the results in the subsequent section with the optimal parameters configuration found on the validation set.

– We evaluate the model’s performance on the test set iteratively for each k-prefixes, i.e., $h d ^ { k } ( \sigma )$ . This is to illustrate the model’s predictive capability given the limited size of prefix, e.g. only single event, as an input. We iteratively compute the performance metric score, such as accuracy, MAE, for each k-prefixes and reports the average results across all prefixes in the following section.

## 5.3 Results

We report the experimental results to assess the performance of <sup>ProcessTrans-</sup> <sup>former</sup> for prediction of the next activity, event time, and remaining time of a running case. Table 2 reports accuracy, F-score, and MAE for nine datasets on three process monitoring tasks. Due to large event logs and time constraints, we do not provide baseline comparison scores for four datasets, namely, BPIC20i, BPIC20d, hospital, and trafic fine logs. For the other five datasets, we report performance comparisons with other related studies. We adopt the baselines scores from the benchmark survey on PBPM as reported in [28].

Table 2. Performance evaluation scores of <sup>ProcessTransformer</sup> for nine event logs for three predictive monitoring tasks.

<table><tr><td rowspan="2"></td><td colspan="2">Next Activity</td><td>Next Event Time</td><td>Remaining Time</td></tr><tr><td colspan="2">Accuracy F-score</td><td>MAE</td><td>MAE</td></tr><tr><td>Helpdesk</td><td>85.63</td><td>0.82</td><td>2.98</td><td>3.72</td></tr><tr><td>BPIC12</td><td>85.20</td><td>0.83</td><td>0.25</td><td>4.60</td></tr><tr><td>BPIC12w</td><td>91.51</td><td>0.91</td><td>0.37</td><td>4.87</td></tr><tr><td>BPIC12cw</td><td>78.48</td><td>0.77</td><td>0.82</td><td>5.14</td></tr><tr><td>BPIC13i</td><td>62.11</td><td>0.60</td><td>0.99</td><td>8.36</td></tr><tr><td>BPIC20d</td><td>86.07</td><td>0.84</td><td>1.22</td><td>2.44</td></tr><tr><td>BPIC20i</td><td>93.35</td><td>0.91</td><td>3.26</td><td>10.68</td></tr><tr><td>Hospital</td><td>85.83</td><td>0.82</td><td>9.33</td><td>44.87</td></tr><tr><td>Traffic fines</td><td>90.00</td><td>0.87</td><td>40.28</td><td>98.24</td></tr></table>

Next Activity Prediction Table 3 reports the (weighted) accuracy for the next activity prediction task on five real-life event logs. Except for BPIC13, <sup>ProcessTransformer</sup> consistently outperforms multiple baselines reported in the literature for the next activity task. Notably, we achieve 86%, 85%, 91% and 78% accuracy scores for Helpdesk, BPI2012, BPI2012w, and BPI2012cw datasets, respectively. The low accuracy of BPIC13 can be attributed to the fewer but lengthy cases in the logs. We also show that on average across all the datasets, <sup>ProcessTransformer</sup> outperforms other approaches.

It is worth noting that we achieve better generalization and performance without performing extensive preprocessing in terms of removing incomplete process traces and 1-sized prefix having a single event only. Importantly, the model utilizes only event prefixes as an input without utilizing any hand-crafted features. These design choices are made to report the realistic analysis and illustrate the powerful learning capabilities of <sup>ProcessTransformer</sup> even with a single event prefix, duplicate activities, and excessively long process traces. Given this, a direct comparison with some of the proposed methods from literature is unjust due to inconsistent data preprocessing and additional input features for learning, which can be missing in event logs in a real-world setting. For instance, [6] utilize additional event attributes, such as timestamp for event label prediction. Similarly, [33,24,15] performed excessive preprocessing on the Helpdesk and BPI2012 datasets in terms of eliminating process traces depending on their number of events and their duration. It results in a predictive model trained and evaluated on an ideal dataset, which does not reflect real-life complex event logs. Furthermore, we do not provide a comparison against [35], as it utilizes additional synthetic data for model training. However, our technique is complementary and can be combined with GANs to improve performance further.

Table 3. Accuracy score (in %) and averaged scores across all datasets for next activity prediction task (Higher is better). The baseline scores for comparison are taken from [28].

<table><tr><td></td><td>Helpdesk</td><td>BPIC12</td><td>BPIC12w</td><td>BPIC12cw</td><td>BPIC13</td><td>Avg.</td></tr><tr><td>Tax et al. [33]</td><td>75.06</td><td>85.20</td><td>84.90</td><td>67.80</td><td>67.50</td><td>76.09</td></tr><tr><td>Khan et al. [15]</td><td>69.13</td><td>82.93</td><td>86.69</td><td>75.91</td><td>64.34</td><td>75.80</td></tr><tr><td>Camargo et al. [4]</td><td>76.51</td><td>83.41</td><td>83.29</td><td>65.19</td><td>68.01</td><td>75.28</td></tr><tr><td>Evermann et al. [10]</td><td>70.07</td><td>60.38</td><td>75.22</td><td>65.38</td><td>68.15</td><td>67.84</td></tr><tr><td>Mauro et al. [6]</td><td>74.77</td><td>84.56</td><td>85.11</td><td>65.01</td><td>71.09</td><td>76.11</td></tr><tr><td>Pasquadibisceglie et al. [24]</td><td>65.84</td><td>82.59</td><td>81.59</td><td>66.14</td><td>31.10</td><td>65.45</td></tr><tr><td>ProcessTransformer</td><td>85.63</td><td>85.20</td><td>91.51</td><td>78.48</td><td>62.11</td><td>80.58</td></tr></table>

Event Time Prediction In Table 4, we present the MAE in days for the event time prediction task against multiple baselines. Our approach has achieved the lowest MAE on average for all considered datasets compared to the previously proposed methods. Besides algorithmic specifications, our approach is diferent to [33,21,15,3] in following aspects. We deal with the event time prediction as an independent task as opposed to the common multi-task approach. This is because the multi-task approach requires dual loss optimization, and there is no guarantee that it will perform better than its single-task counterpart [41]. We also create additional temporal features (see Definition 4 in Section 4) to equip model with a sense of events’ duration. To summaries, <sup>ProcessTransformer</sup> obtains MAE of 2.98, 0.25, 0.37, 0.82 and 0.99 for Helpdesk, BPI2012, BPI2012w, BPI2012cw and BPIC13 datasets, respectively.

Table 4. MAE (in days) and averaged scores across all datasets for event time prediction task (Lower is better). The baseline scores for comparison are taken from [28].

<table><tr><td></td><td colspan="6">Helpdesk BPIC12 BPIC12w BPIC12cw BPIC13 Avg.</td></tr><tr><td>Tax et al. [33]</td><td>5.77</td><td>0.31</td><td>0.50</td><td>1.20</td><td>0.47</td><td>1.65</td></tr><tr><td>Khan et al. [15]</td><td>6.33</td><td>0.31</td><td>0.50</td><td>1.32</td><td>0.55</td><td>1.80</td></tr><tr><td>PROCESSTRANSFORMER</td><td>2.98</td><td>0.25</td><td>0.37</td><td>0.82</td><td>0.99</td><td>1.08</td></tr></table>

Remaining Time Prediction We use the same model architecture and temporal features for the remaining time prediction problem as for event time prediction. Table 5 reports the MAE scores averaged across all the prefixes. Our approach outperforms previous methods on average across considered datasets (see the last column of Table 5). Specifically, we obtain MAE of 3.72, 4.60, 4.87, 5.14, and 8.36 for Helpdesk, BPI2012 BPI2012w, and BPI2012cw and BPIC3 datasets, respectively. We note that, the performance diference between Tax et al [33] and Navarin et al [20] is due to diference in encoding techniques and temporal features used, as they both use LSTM as a base model for learning the predictive task.

Table 5. MAE (in days) and averaged scores across all datasets for remaining time prediction task (Lower is better). The baseline scores for comparison are taken from [28].

<table><tr><td></td><td>Helpdesk</td><td>BPIC12</td><td>BPIC12w</td><td>BPIC12cw</td><td>BPIC13</td><td>Avg.</td></tr><tr><td>Tax et al. [33]</td><td>71.50</td><td>330.61</td><td>387.81</td><td>210.16</td><td>38.41</td><td>207.70</td></tr><tr><td>Camargo et al. [4]</td><td>11.15</td><td>30.56</td><td>32.03</td><td>7.97</td><td>260.64</td><td>68.47</td></tr><tr><td>Navarin et al. [20]</td><td>10.38</td><td>6.13</td><td>6.63</td><td>6.48</td><td>2.97</td><td>6.52</td></tr><tr><td>PROCESS TRANSFORMER</td><td>3.72</td><td>4.60</td><td>4.87</td><td>5.14</td><td>8.36</td><td>5.33</td></tr></table>

## 6 Conclusion

The main contribution of our study is a <sup>ProcessTransformer</sup> approach for learning high-level representations directly from sequential event logs data with minimal preprocessing. We evaluate our approach for the next activity, event time, and remaining time prediction tasks on nine real-life event logs. We show that <sup>ProcessTransformer</sup> can capture long-range dependencies without the explicit need of recurrence as LSTM-based models. Our approach outperforms several existing baselines in experimental evaluations. Specifically, we achieve an average of above 80% accuracy on considered datasets for the next activity prediction task while solely using the activity prefix as an input for the model. Similarly, our approach obtains an average MAE of 1.08 and 5.33 for predicting the next event time and completion time of a running case. Notably, we use minimal data preprocessing and features as an input to illustrate the learning capability of <sup>ProcessTransformer</sup> . This is also to emphasis that the <sup>Pro-</sup> <sup>cessTransformer</sup> can provide optimal predictive performance for real event logs even when additional attributes, such as resources, roles, and others, are missing in the dataset.

The future work seeks to study how the learned representations can be used for other tasks of interest, including similar trace retrieval, activity recommendations, and process outcome prediction. Another future avenue is to evaluate proposed <sup>ProcessTransformer</sup> with event logs having not only prolonged but largely unique process activity space.

Reproducibility The source code and supplementary material is publicly available at https://github.com/Zaharah/processtransformer.

## References

1. Van der Aalst, W.M., Schonenberg, M.H., Song, M.: Time prediction based on process mining. Information systems 36(2) (2011)

2. Bahdanau, D., Cho, K., Bengio, Y.: Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014)

3. B¨ohmer, K., Rinderle-Ma, S.: LoGo: Combining local and global techniques for predictive business process monitoring. In: Proc. of CAiSE. Springer (2020)

4. Camargo, M., Dumas, M., Gonz´alez-Rojas, O.: Learning accurate LSTM models of business processes. In: Proc. of BPM. Springer (2019)

5. Conforti, R., De Leoni, M., La Rosa, M., Van Der Aalst, W.M.: Supporting riskinformed decisions during business process execution. In: Proc. of CAiSE (2013)

6. Di Mauro, N., Appice, A., Basile, T.M.: Activity prediction of business process instances with inception CNN models. In: Proc. of AIIA. Springer (2019)

7. van Dongen, B.: BPI Challenge 2012 (Apr 2012). https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

8. van Dongen, B.: BPI Challenge 2020: Domestic Declarations (Mar 2020). https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5

9. van Dongen, B.: BPI Challenge 2020: International Declarations (Mar 2020). https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5

10. Evermann, J., Rehse, J.R., Fettke, P.: A deep learning approach for predicting process behaviour at runtime. In: Proc. of BPM. pp. 327–338. Springer (2016)

11. Folino, F., Guarascio, M., Pontieri, L.: Context-aware predictions on business processes: an ensemble-based solution. In: Proc. of NFMCP. Springer (2012)

12. Guo, C., Berkhahn, F.: Entity embeddings of categorical variables. arXiv preprint arXiv:1604.06737 (2016)

13. Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural computation 9(8) (1997)

14. Kang, B., Kim, D., Kang, S.H.: Periodic performance prediction for real-time business process monitoring. Industrial Management & Data Systems (2012)

15. Khan, A., Le, H., Do, K., Tran, T., Ghose, A., Dam, H., Sindhgatta, R.: Memoryaugmented neural networks for predictive process analytics. arXiv preprint arXiv:1802.00938 (2018)

16. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

17. de Leoni, M.M., Mannhardt, F.: Road trafic fine management process (Feb 2015). https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

18. Lin, L., Wen, L., Wang, J.: Mm-pred: A deep predictive model for multi-attribute event sequence. In: Proc. of ICDM. SIAM (2019)

19. Mannhardt, F.: Hospital billing - event log (Aug 2017). https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741

20. Navarin, N., Vincenzi, B., Polato, M., Sperduti, A.: LSTM networks for data-aware remaining time prediction of business process instances. In: Proc. of SSCI. IEEE (2017)

21. Nguyen, A., Chatterjee, S., Weinzierl, S., Schwinn, L., Matzner, M., Eskofier, B.: Time Matters: Time-Aware LSTMs for Predictive Business Process Monitoring. arXiv preprint arXiv:2010.00889 (2020)

22. Pandey, S., Nepal, S., Chen, S.: A test-bed for the evaluation of business process prediction techniques. In: 7th International Conference on Collaborative Computing: Networking, Applications and Worksharing. IEEE (2011)

23. Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q.N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., Fern´andez, R.: The LAMBADA dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031 (2016)

24. Pasquadibisceglie, V., Appice, A., Castellano, G., Malerba, D.: Using convolutional neural networks for predictive process analytics. In: Proc. of ICPM. IEEE (2019)

25. Pauwels, S., Calders, T.: Bayesian network based predictions of business processes. In: Proc. of BPM. Springer (2020)

26. Polato, M.: Dataset belonging to the help desk log of an italian company (Jul 2017). https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb

27. Polato, M., Sperduti, A., Burattin, A., de Leoni, M.: Time and activity sequence prediction of business process instances. Computing 100(9) (2018)

28. Rama-Maneiro, E., Vidal, J.C., Lama, M.: Deep learning for predictive business process monitoring: Review and benchmark. preprint arXiv:2009.13251 (2020)

29. Rogge-Solti, A., Weske, M.: Prediction of remaining service execution time using stochastic petri nets with arbitrary firing delays. In: Proc. of ICSOC. Springer (2013)

30. Sokolova, M., Lapalme, G.: A systematic analysis of performance measures for classification tasks. Information processing & management 45(4), 427–437 (2009)

31. Steeman, W.: BPI Challenge 2013, incidents (Apr 2013). https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee

32. Sutskever, I., Vinyals, O., Le, Q.V.: Sequence to sequence learning with neural networks. arXiv preprint arXiv:1409.3215 (2014)

33. Tax, N., Verenich, I., La Rosa, M., Dumas, M.: Predictive business process monitoring with LSTM neural networks. In: Proc. of CAiSE. Springer (2017)

34. Taymouri, F., La Rosa, M.: Encoder-decoder generative adversarial nets for suffix generation and remaining time predication of business process models. arXiv preprint arXiv:2007.16030 (2020)

35. Taymouri, F., La Rosa, M., Erfani, S., Bozorgi, Z.D., Verenich, I.: Predictive business process monitoring via generative adversarial nets: The case of next event prediction. arXiv preprint arXiv:2003.11268 (2020)

36. Teinemaa, I., Dumas, M., Rosa, M.L., Maggi, F.M.: Outcome-oriented predictive process monitoring: Review and benchmark. TKDD 13(2) (2019)

37. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Proc. of NeuroIPS (2017)

38. Wang, J., Yu, D., Liu, C., Sun, X.: Outcome-oriented predictive process monitoring with attention-based bidirectional LSTM neural networks. In: Proc. of ICWS. IEEE (2019)

39. Weinzierl, S., Zilker, S., Brunk, J., Revoredo, K., Nguyen, A., Matzner, M., Becker, J., Eskofier, B.: An empirical comparison of deep-neural-network architectures for next activity prediction using context-enriched process event logs. arXiv preprint arXiv:2005.01194 (2020)

40. Wolf, T., Chaumond, J., Debut, L., Sanh, V., Delangue, C., Moi, A., Cistac, P., Funtowicz, M., Davison, J., Shleifer, S., et al.: Transformers: State-of-the-art natural language processing. In: Proc. of EMNLP (2020)

41. Zhang, Y., Yang, Q.: A survey on multi-task learning. arXiv preprint arXiv:1707.08114 (2017)