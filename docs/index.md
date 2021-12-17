# Word/Character-level tokenizers in Chinese-English Neural Machine Translation

## Abstract:
Machine Translation is one of the hot topic recently in NLP, especially during the era of neural models. In fact, with the rise of Transformer-based models to keep its top-rank performance, Transformer-based models are being used more and more in many language-related tasks. Within this field, we noticed that the Chinese language is different than the ones using Latin-based alphabet. In Chinese, character is a different concept other than letter or word. If anything, chinese characters are like bound morphemes. Chinese characters do not constitute an alphabet or a compact syllabary. Rather, the writing system is roughly logosyllabic; that is, a character generally represents one syllable of spoken Chinese and may be a word on its own or a part of a polysyllabic word. Understanding this issue, we want to investigate how a Transformer-based model would perfom when dealing with the Chinese-English neural machine translation task.

## Related Work:


## Experiments:

### Experiment setup:

#### Dataset:
- Examples data:

- Analysis of data:

Number of words (EN) | Number of unique words (EN) | Number of words (CN) | Number of unique words (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 7640775 | 122432 

Number of words (EN) | Number of unique words (EN) | Number of characters (CN) | Number of unique character (CN) 
--- | --- | --- |--- 
8080917 | 70761 | 13406447 | 4723

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
