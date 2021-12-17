# Word/Character-level tokenizers in Chinese-English Neural Machine Translation

## Abstract:

## Related Work:

## Experiments:

### Experiment setup:

#### Dataset:
- Analysis of data:
| Number of words (EN) | Number of unique words (EN) | Number of words (CN) | Number of unique words (CN) |
| --- | ----------- | ----------- | ----------- |
| Header | Title | ----------- | ----------- |
| Paragraph | Text | ----------- | ----------- |

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
- Examples

## Conclusion:


## References:
