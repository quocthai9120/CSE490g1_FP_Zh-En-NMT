# Word/Character-level tokenizers in Chinese-English Neural Machine Translation

## Abstract:
Machine Translation is one of the hot topic recently in NLP, especially during the era of neural models. In fact, with the rise of Transformer-based models to keep its top-rank performance, Transformer-based models are being used more and more in many language-related tasks. Within this field, we noticed that the Chinese language is different than the ones using Latin-based alphabet. In Chinese, character is a different concept other than letter or word. If anything, chinese characters are like bound morphemes. Chinese characters do not constitute an alphabet or a compact syllabary. Rather, the writing system is roughly logosyllabic; that is, a character generally represents one syllable of spoken Chinese and may be a word on its own or a part of a polysyllabic word. Understanding this issue, we want to investigate how a Transformer-based model would perfom when dealing with the Chinese-English neural machine translation task.

## Related Work:
### Model architecture
Machine Translation has a long history of development. Previously, researchers came up with Recurrent networks to extract the sequential information within sentence (cite). Encoder-Decoder arrives to give us a better way to structure our architecture in which we can condense information using embedding then use these latent features for decoding (translating to another language) (cite). Nowadays, researchers have shown how powerful transformer models are as these models could learn the relationship of tokens within sentences while keeping track of sequential information (cite).

When looking for ways to break our sentences into tokens, we have found several useful tokenizers online.
1. Spacy, which gives us word-level tokens.
2. BERT Chinese, which gives us character-level tokens.

### Dataset:
- Examples data: 
        
  
      And much of the anger that drove people to the streets, led countries to the point of collapse, and drove millions from their homes was motivated by a desire for clear rights, including those protecting property.
      而让人民涌向街头，让国家走向崩溃，让数百万人走出家园的愤怒背后正是对明晰的权利的渴望，包括保护财产的权利。
      The pessimists claim that this is becoming harder and more expensive; the optimists hold that the law will remain valid, with chips moving to three dimensions.
      悲观主义者认为增长将变得更为困难和昂贵；乐观主义者则认为这一定律将随着芯片向3D阵列发展而继续有效。
      Russia and China today are united not only by their energy deals, but also by both countries’ conviction that their time has come, and that the outside world needs them more than they need the outside world, particularly the US.
      今天，将俄国和中国联系在一起的已经不仅仅是两国的能源协议，这两个国家都认定，属于他们的时代已经到来。 世界需要他们更甚于他们需要世界，而美国尤其如此。
      Yet that isn’t helping the PD.
      但这并没有对PD起到帮助作用。
- Analysis of data:

![histogram of source language](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/source_language_hist.png?raw=true)
![histogram of target language](https://github.com/quocthai9120/cse490g1_zh_en_nmt/blob/main/docs/graphs/target_language_hist.png?raw=true)

Number of words (EN) | Number of unique words (EN) | Number of words (CN) | Number of unique words (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 7640775 | 122432 

Number of words (EN) | Number of unique words (EN) | Number of characters (CN) | Number of unique character (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 13406447 | 4723

## Experiments:

### Experiment setup:

#### Preprocessing data:

### Tokenizer:

#### Word-level tokenizer:

- We experimented a word-level tokenizer from Spacy to tokenize both Chinese and English sentences.
- Vocab size:
  +. Chinese: 108342
  +. English: 65583
- How do you process your tokenizing process:
  +. Define special symnbols and indx
  +. Create a vocab to store all the mapping from word to indexes
- Perform tokenize for EN and CN

#### Character-level tokenizer
- We experimented a word-level tokenizer from Spacy to tokenize both Chinese and English sentences.
- Vocab size:
  +. Chinese: ...
  +. English: ...
- How do you process your tokenizing process:
  +. Applying BERT tokenizer for each sentences

### Model design:
#### Model architecture:
- Transformer-based
- Linear layer on top
- Mention how many heads, how many emb layers
- Add a graph of our model

#### Hyperparameters:
- ....
- Gradient accumulation

## Evaluation:
- BLEU score
  +. Make plot for different n-grams
  +. Short sentences / medium sentences / long sentences
  
- Examples

## Conclusion:


## References:
