---
draft: false
title: "Semantic Search in Layman Terms"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1472512946974-cc09a294e210?q=80&w=2746&auto=format&fit=crop&w=430&h=240",
    alt: "semantic search"
}
publishDate: "2023-05-17 00:00"
category: "Concepts"
author: "Praveen"
tags: [machinelearning, vectorsearch, starter]
---

If you are someone like me who is hearing about semantic search, vectors and embeddings after LLM(Large Language Model) was launched and finds these terms confusing then I hope this blog brings some clarity to you.

## What is Semantic Search
Semantic search in Natural Language Processing (NLP) refers to the process of understanding the **meaning** or **intent** behind a user's search query and retrieving relevant information based on that understanding. Unlike traditional keyword-based search, which matches queries to documents based on exact word matches, semantic search aims to comprehend the context and semantics of the query to generate more **accurate** and **contextually relevant** results.

The next question is how to make computers understand the semantic information... Humans have very high cognitive capabilities so they can easily understand semantics in multiple languages but to make a computer understand semantics is challenging.

In this blog, we are going to see how semantic information is understood using vectors/embeddings. In my previous blog, I have shown how [CountVectorizer & TFIDF](https://dev.to/praveenr2998/countvectorizer-vs-tfidf-logistic-regression-3heb) works now we are going to see an even more advanced yet simple and easy way to do semantic search

# What is a vector
Mathematically a vector is a value which has both **magnitude** and **direction**.


![Single Dimensional Vector](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/c4rpk97gswr4nj86x3iv.gif)
Here vectors A,B,C,D have magnitudes 4, 2 and A,B,D have same direction but C has a different direction. These are **single dimensional** vectors.

![Multi Dimensional Vector](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/6pxa33reehkay8gch0y2.png)
In mathematics unlike in physics, there could be **n** dimensions for a vector and these are called **multi-dimensional** vectors(each arrow in the above figure is a dimension). The **all-MiniLM-L6-v2** model that we are going to use in this blog generates a vector with dimension **384**. The information stored in these dimensions is used to find semantic similarity.

# Sentence Transformers
SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. It provides easy methods to compute **embeddings** (dense vector representations) for sentences, paragraphs and images.

Now we are going to see how to generate embeddings and do a semantic search using a pre-trained model from sentence transformers.


## Pretrained Sentence Transformer Model - all-MiniLM-L6-v2
We have a Python library to access the model
```python
pip install -U sentence-transformers
```
We are going to use the **all-MiniLM-L6-v2** model which is a lightweight yet powerful model.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```
Now we can take a few question-answer sentences and generate embeddings from them and do a semantic search on them.

```python
# Q&A sentences
question_answers = [
           "Q : What is this software used for? A : This software is used to handle you finances and provide useful suggestion",
           "Q : How much does it cost per year? A : It costs 5000 rupees per year",
           "Q : Is there a premium version available? A : Yes it is available for a cost of 7000 rupees per year",
           "Q : Why should I choose this rather than product Y? A : Our product outperforms in W and Z"  
          ]

#Sentences are encoded by calling model.encode()
question_answer_embeddings = model.encode(question_answers, convert_to_tensor=True)
```
The **encode** function will generate embeddings and further, we are converting the embeddings into a **pytorch tensor**.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/i42p41gsugrzvc5wx274.png)
That was easy !!!!

Now we will ask a question and find semantically relevant content from the embeddings generated.
```python
question = ['Can you explain the use of this software']
question_embeddings = model.encode(question, convert_to_tensor=True)
```  
We should also generate embedding for our question...

```python
from sentence_transformers.util import semantic_search
hits = semantic_search(question_embeddings, question_answer_embeddings, top_k=1)
```
We are using a utility function called **semantic_search** which internally uses [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) by default to find the **similarity** between the two embeddings and returns a **similarity score**, you can also use any other metric for comparing the vectors like **dot product**.


![Hits](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/kyi9i0mrkwr9pe18evyv.png)

```python
print([question_answers[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
```

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/mwirzr7pzqbnha7bxtmd.png)
 
```python
question = ['How much do you charge?']
question_embeddings = model.encode(question, convert_to_tensor=True)

from sentence_transformers.util import semantic_search
hits = semantic_search(question_embeddings, question_answer_embeddings, top_k=1)

print([question_answers[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
```

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jokiicrxvkoy7x90sif9.png)

You could observe from the above examples that the question asked is not exactly matching to any input in question_answers but we are able to find the one that closely matches our input.

There are many other models to generate even more powerful embeddings and the quality of embeddings is directly proportional to the semantic similarity.
Happy Learning :))
