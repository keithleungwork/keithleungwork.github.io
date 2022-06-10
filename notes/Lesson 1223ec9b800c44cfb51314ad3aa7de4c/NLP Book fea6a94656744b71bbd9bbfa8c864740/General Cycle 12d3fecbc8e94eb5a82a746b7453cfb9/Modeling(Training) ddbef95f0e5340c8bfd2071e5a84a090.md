# Modeling(Training)

## Handcraft Approach:

At the start, data size is small, we should start with simple methods or rule based system.

- start with heuristics method
I.e. by trial & error OR loosely defined rules
    - E.g. An explicit blacklist in email spam task.
    - E.g.2 In E-commerce, sorting & recommend by the top number of purchased items(or its category)
- Another example is regular expression
    - lib: `Stanford NLP’s TokensRegex` , `spaCy’s rule based matches`

When data grows, switch to/combine with ML or DL model.

There are 2 popular ways:

- Create features from heuristics, to train/inference model
- Use heuristics in particular case and bypass model 
(I.e. the heuristics decide it during pre-processing)

## Using  NLP service providers as approach:

- Google Cloud Natural Language
- Amazon Comprehend
- Microsoft Azure Cognitive Services
- IBM Watson Natural Language Understanding

> These can also be used as a reference to see if your model performs well enough. If not, then you can use the NLP service directly.
> 

## Model ensembling & stacking

- Common way is to NOT do everything by 1 model. Instead, use multiple model to do each specific task.
- Ensembling: model in parallel
- Stacking: model in series

## Apply Heuristic after model inference

- it is a good practice to apply rule based / heuristic test after model  final output, as a safe test to ensure the model does not make huge mistake

## Summary of How to choose approach based on Data size

![3A8D6C13-4191-4C8F-8A5F-30095966C5E1.jpeg](Modeling(Training)%20ddbef95f0e5340c8bfd2071e5a84a090/3A8D6C13-4191-4C8F-8A5F-30095966C5E1.jpeg)