# Text similarity/comparison

## Native pacakage - SequenceMatcher

[https://docs.python.org/3/library/difflib.html](https://docs.python.org/3/library/difflib.html)

There are are many are different string metrics like [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance)
, [Damerau-Levenshtein](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
, [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)
, [Jaro-Winkler](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance)
 and [Strike a match](http://www.catalysoft.com/articles/StrikeAMatch.html)
.

## Levenshtein

- much faster than sequenceMatcher

## **Locality-sensitive hashing**

- [https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134](https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134)

## Elasticsearch

- It supports fuzzy search - which calculates Levenshtein distance
    - [https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-fuzzy-query.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-fuzzy-query.html)
- Text similarity search with vectors
    - [https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch](https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch)
- Advanced usage - Text similarity with TF models and Elastic search
    - [https://www.ulam.io/blog/text-similarity-search-using-elasticsearch-and-python](https://www.ulam.io/blog/text-similarity-search-using-elasticsearch-and-python)
        - Not training a model, but use a pre-trained model to get text embeddings