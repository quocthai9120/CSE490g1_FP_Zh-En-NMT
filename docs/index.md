# Word-level Tokenized Transformer-based Model in Chinese-English Neural Machine Translation

## Abstract:
Machine Translation is one of the hot topic recently in NLP, especially during the era of neural models. In fact, with the rise of Transformer-based models to keep its top-rank performance, Transformer-based models are being used more and more in many language-related tasks. Within this field, we noticed that the Chinese language is different than the ones using Latin-based alphabet. In Chinese, character is a different concept other than letter or word. If anything, chinese characters are like bound morphemes. Chinese characters do not constitute an alphabet or a compact syllabary. Rather, the writing system is roughly logosyllabic; that is, a character generally represents one syllable of spoken Chinese and may be a word on its own or a part of a polysyllabic word. Understanding this issue, we want to investigate how a Transformer-based model would perfom when dealing with the Chinese-English neural machine translation task.

We start our experiment by analyzing our dataset to determine the architectures and pre-processing directions we should do. We then perform pre-processing steps to clean up our data and perform tokenization. We implement a "RNN encoder + Attention RNN decoder" architecture as a baseline model. We then create a Transformer-based model (our proposed architecture in this report). Finally, we propose a future work of using a character-level tokenization with Transformer-based model. All of our codes are included inside our Git repo for reproducing and further researching purposes. Readers can access our Git repo by following this URL: [ZH-EN Neural Machine Translation - Github](https://github.com/quocthai9120/cse490g1_zh_en_nmt). 

## Related Work:
### Model architecture
Machine Translation has a long history of development. Previously, researchers came up with Recurrent networks to extract the sequential information within sentence (Neco, R. P., & Forcada, M. L., 1997). Encoder-Decoder arrives to give us a better way to structure our architecture in which we can condense information using embedding then use these latent features for decoding (translating to another language) (Sutskever, I., Vinyals, O., & Le, Q. V., 2014). Nowadays, researchers have shown how powerful Transformer-based models are as these models could learn the relationship of tokens within sentences while keeping track of sequential information (Ashish Vaswani, et al., 2017).

### Dataset:
We use the dataset from WMT21 (https://www.statmt.org/wmt21/translation-task.html). The data sources are new-commentary corpus which are released for the WMT series of shared task. The one we are using is a parallel corpus containing 313674 parallel Chinese-English sentences. It can be found and downloaded using this following url: https://data.statmt.org/news-commentary/v16/.

Below, we show several examples of the dataset we are using:        
  
      EN: And much of the anger that drove people to the streets, led countries to the point of collapse, and drove millions from their homes was motivated by a desire for clear rights, including those protecting property.
      CN: 而让人民涌向街头，让国家走向崩溃，让数百万人走出家园的愤怒背后正是对明晰的权利的渴望，包括保护财产的权利。
 
      EN: The pessimists claim that this is becoming harder and more expensive; the optimists hold that the law will remain valid, with chips moving to three dimensions.
      CN: 悲观主义者认为增长将变得更为困难和昂贵；乐观主义者则认为这一定律将随着芯片向3D阵列发展而继续有效。
      
      EN: Russia and China today are united not only by their energy deals, but also by both countries’ conviction that their time has come, and that the outside world needs them more than they need the outside world, particularly the US.
      CN: 今天，将俄国和中国联系在一起的已经不仅仅是两国的能源协议，这两个国家都认定，属于他们的时代已经到来。 世界需要他们更甚于他们需要世界，而美国尤其如此。
      
      EN: Yet that isn’t helping the PD.
      CN: 但这并没有对PD起到帮助作用。
      
Before diving into constructing our model, we have done several analysis on the dataset. After analyzing, we noticed that Chinese sentences in our dataset are long on average. Particularly, the average number of words in a Chinese sentence is around 22 words. The English sentences are also really long on average, too, with an average of 25 words / sentence. When breaking down to character-level for Chinese, we can also see that it has an average of 50 characters per sentence.

![histogram of source language character level](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/chinese_character_level.png?raw=true)
![histogram of source language word level](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/chinese_word_level.png?raw=true)
![histogram of target language](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/english_word_level.png?raw=true)

Besides, the number of unique words in Chinese are over 100 thousands, while the number of unique characters in Chinese are just around 5000. More details would be summarized below:

Number of words (EN) | Number of unique words (EN) | Number of words (CN) | Number of unique words (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 7640775 | 122432 

Number of words (EN) | Number of unique words (EN) | Number of characters (CN) | Number of unique character (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 13406447 | 4723

Analyzing our data gives us the thought that we need to create a model that could deal with complex sentences so that it could learn deep features from our complex dataset.

## Experiments:
Our final model is a Transformer-based Sequence-to-Sequence model with word-level tokenized sentences. The progress included cleaning the data, tokenizing, fitting the format to train, training the model, and evaluating the model. 

### Preprocessing data:
The first step of our experiment is to make the data to have the right format to fit into our model. We would do that as follow:

#### Cleaning the data:
We noticed that the dataset has consecutive sentences that are extracted from the same articles. To make the model generalize better, we shuffled our data before using it. Then, we just need to split our dataset into training dataset (80%),validation dataset(10%) and testing dataset (10%).

#### Tokenizer:
We use a word-level tokenizer from Spacy to tokenize both Chinese and English sentences. A brief summary of the tokenizer is: (1) The Chinese tokenizer that we are using has a vocabulary size of 108,342; (2) The English Tokenizer that we are using has a vocabulary size of 65,583.

We implemented the progress of tokenizing our sentences as follow:
  1. Add the sentences with a beginning tag (\<bos\>) and an ending tag (\<eos\>).
  2. Pad the sentence to the maximum length of the current batch. That is, the length of each batch is equal to the length of the longest sentence within the batch.
  3. Create a vocabulary map to map from words to tokens.
  4. Generate attention and padding mask to prevent leftward information flow in the decoder to preserve the auto-regressive property of the model.
  5. Create a transformation function to transform input sentences within a batch into tokenized tensors within that same batch. Particularly, each batch's tokenized tensor would be created by concating a tensor of beginning tags, a tensor of token ids for each tokenized sentence within that batch, and a tensor of ending tags.

### Model design:
#### Baseline:
We use a "RNN encoder and Attention RNN decoder" as our baseline. We train our model using SGD optimizer and set the learning rate to 0.01. Training until convergence, we got a BLEU-score of around 0.08. Since our paper's attention is not this part, we would not go into detail. Instead, we would leave the implementation notebook here for readers to investigate if they have interest. The url to the baseline notebook is left here: [RNN baseline](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/zh_en_machine_translation_baseline_rnn.ipynb).

#### Model architecture:
- After looking at our anlayzed data, we decided to create a transformer-based model to deal with the complexities within a single sentence. Particularly, since our sentences are long on average, we want to have 8 self-attention heads and an embedding size of 512 to "learn" the complex distanct dependency of components within a sentence.
  
- On the other hand, we also noticed that our dataset is not that big (particularly, we only have over 300 thousands instances in our dataset), so we decided to make our model to have only 3 encoder layers and 3 decoder layers (plus one linear output layer on top) to avoid overfitting.
  
Besides, we create a visualization to demonstrate how our model would be end-to-end. Note that, for the Transformer block, we use the exact same architecture as the one shown in the original paper with our modifications mentioned above.

![Model architecture](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/model%20architecture.png?raw=true)

The notebook for our proposed model can be accessed through the following url: [Proposed model notebook](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/zh_en_machine_translation_word_level_transformer.ipynb).

## Training
We train our model using Google Collaborative GPU. Each training epoch takes us around 1 hour. After several times of training and finetuning hyperparameters, we end up with the following setup:
  
### Hyperparameters:
We train our model described above with the following hyperparameters:
- We use Adam optimizer with learning rate to be 0.0001 as we heuristically believe this would perform better than using SGD.
- We decide to use the Cross-Entropy loss. This is because we want to "learn" the probability of each word to be predicted from the targeted language, and Cross-Entropy reflects that good.
- Finally, we use a batch size of 4 with gradient accumulation step of 64. This is because we are training on the Google Collaborate GPU, we cannot make our batch size to be bigger. However, traihing with just a batch size 4 would make our model inconsitent in learning. For that reason, using gradient accumulation, we make our model to update its parameters every 64 accumulation steps. In the other words, we make our model to train with a batch size of 256. This makes the training progress become faster and stedier.

Training our model for 6 epochs, we achieved a training loss of around 2.596 and a validation loss of around 2.780. We consider this would be a good result at this point.
![histogram of target language](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/training_graph.png?raw=true)  

## Evaluation:
After finishing training, we evaluate our model performance on an unseen test set. The test loss is around 2.671.
  
Moreover, since we are performance translation task, we want to have a better metric for evaluating the translation performance. We decided that BLEU score is a good candidate to be used here. A reminder of BLEU score is that:

  
![BLEU score formula](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/formula.png?raw=true)  

  
Using our trained model with the test set, we achieved a BLEU-score of 0.167079.
  
We also bring this to the next level by evaluating our model performance toward short sentences, medium sentences, and long sentences. Particularly, we compute our BLEU-score in these categories and have the following result:
        
Average target sentence length | Short sentences (5 words) | medium sentences (25 words) | long sentences (58 words)
--- | --- | --- |--- 
BLEU | 0.1316437 | 0.1730123877 | 0.14096469
  
  
Below are some translations generated by our final model:
        
        Source sentence: 危机爆发前，欧洲似乎是首度成功实现政治一体化平衡状态可能性最大的候选人。
        Groud truth translation: Before the crisis, Europe looked like the most likely candidate to make a successful transition to the first equilibrium – greater political unification.
        Our translation: Before the crisis, Europe seems to be the most likely candidate for success in achieving political integration.
        
        Source sentence: 尽管希腊救助计划已经完成，但欧元危机并未真正落幕，尤其是意大利可能成为风险的主要来源
        Groud truth translation: The euro crisis is not truly over, despite the completion of Greece’s bailout program, with Italy, in particular, representing a major source of risk.
        Our translation: Although the Greek rescue program has been completed, the euro crisis did not end, especially Italy may become the main source of risk.
        
        Source sentence: “重启”与欧洲后院各国的关系
        Ground truth translation: A “Reset” Button for Europe’s Backyard.
        Our translation: “ reset ” relations with Europe ’s neighbors.
        
Looking at the translations, we can see the pattern that our model works best for medium length sentences with around 23 **words**, it would still give fairly good translations for shorter and longer sentences. The reason that makes our model works best for medium length sentences would be due to the fact that for shorter sentences, BLEU-score cannot reflect synonyms as good translation, and for longer sentences, it may predict more wrong words.

## Conclusion:
Using transformer model allows us to learn more of long distance dependancies comparing to RNN-based model. Even though this only improves our performance BLEU-score by 0.08, evaluating the generated output by human lets us know that our model learns much better with the Transformer-based model. This leads to a conclusion that for translating long and complex sentences from a complex language like Chinese, using a Transformer-based model is a great approach to follow.

For our future work, we are experimenting using character-level tokenization with Transformer-based model. However, this might requires a significant amount of modification, so we would not include a final version of that in this project. However, we would leave the url to our most recent notebook of that here: [character-level Transformer-based model](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/zh-en-machine-translation-char-level.ipynb).

## Demo:
Below, we are showing a demo of how to use our trained model for translation. Simply, you can just load the trained model and use that trained model for translation as the following video (click on the thumbnail to see the video):
[![Demo Translation video](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/model%20architecture.png)](https://youtu.be/-9Wi0wKO9yA)

In the video, we have done the following steps:
  1. Load the trained model
  2. Run the code of implementing translation method
  3. Translate the first 10 sentences in our test set
  4. Translate a custom sentence inputted by user

## References:
1. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://aclanthology.org/N19-1423/)

2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all you need.](https://arxiv.org/abs/1706.03762)

3.  Neco, R. P., & Forcada, M. L. (1997, June). Asynchronous translations with recurrent neural nets. In Neural Networks, 1997., International Conference on (Vol. 4, pp. 2535-2540). IEEE.

4. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems(pp. 3104-3112).
