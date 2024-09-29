---
draft: false
title: "My Embeddings Stay Close To Each Other, What About Yours?"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?q=80&w=2832&auto=format&fit=crop&w=430&h=240",
    alt: "embeddings"
}
publishDate: "2024-02-29 15:39"
category: "Concepts"
author: "Praveen"
tags: [machinelearning, embeddings, vectorsearch]
---

This blog will help you generate embeddings for your datasets such that semantically related sentences stay close to each other in other words, this blog will help you fine-tune commonly available SBERT(Sentence BERT) models in [hugging face](https://huggingface.co/sentence-transformers) using your dataset.

## LITTLE BACKGROUND ABOUT SBERT
Sentence BERT was first introduced in the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). In this paper, the authors have proposed a modification of the pre-trained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity.

This blog is not about how SBERT works but rather how to finetune a pre-trained SBERT, so let's go ahead.

## WHY FINETUNE
Sometimes when you try to retrieve some information using any distance metric like cosine similarity the retriever might fetch unintended information, the reason being the unintended information is closer to your query in vector space.

![2D SPACE](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/si7iv68kal9mq42nydkf.jpg)
In the above image your question vector and irrelevant vector are close to each and why does this happen ???
A few reasons might be

- Wrong choice of embedding model - The model might be trained on a dataset from a different domain.

- The terms or words that you use might be unseen during model training

## SO WHAT'S THE SOLUTION
If you find that your use case has some unseen words or you have better datasets which you believe could make the model generate quality embeddings you could go for **fine-tuning**.

## FINE-TUNING SENTENCE BERT FROM HUGGING FACE
We are going to use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from [hugging face](https://huggingface.co/).

### Required Libraries
```python
pip3 install torch
pip3 install pandas
pip3 install -U sentence-transformers
```
### Little Bit Of Clarity
By finetuning we mean to ask the model to consider the pair of sentences that we send as training data points to be close to each other, there are several ways to **organize** your training data and a table explaining it is given below

![Table](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/rim6vaqca75wib2ukl70.png)
[image credits](https://huggingface.co/blog/how-to-train-sentence-transformers)

In this blog, we are going to use a **pair of positive sentences without label** for each training data point and the sentence pair denotes closely related sentences. The corresponding loss function would be [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss)

## TRAINING
```python
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
```
```python
class trainSBERT:
    def prepare_training_data(self, source_sentence_list, target_sentence_list):
        """
        Each training data point must have 2 two similar sentences inside a list
        Eg - [sentence 1, sentence 2]

        INPUT
        source_sentence_list - List : All source sentences
        target_sentence_list - List : All target sentences

        RETURNS
        train_dataloader - Pytorch dataloader object
        """
        train_data_list = []
        for source, target in zip(source_sentence_list, target_sentence_list):
            print(source, target)
            train_data_list.append(InputExample(texts=[source, target]))
       
        train_dataloader = DataLoader(train_data_list, shuffle=True, batch_size=64)
        return train_dataloader
   
    def train_sbert(self, model_name_list, n_epochs, source_sentence_list, target_sentence_list, path_to_save_model):
        """
        Used to train various sentence bert model

        INPUT
        model_name_list - List : List of model names from hugging face to be trained
        n_epochs - Int : Epochs to be trained for
        source_sentence_list - List : All source sentences
        target_sentence_list - List : All target sentences
        path_to_save_model - String : Path to save trained model

        RETURNS
        None
        """
        train_dataloader = self.prepare_training_data(source_sentence_list, target_sentence_list)
        for model_name in model_name_list:
            sbert_model = SentenceTransformer(model_name)
           
            train_loss = losses.MultipleNegativesRankingLoss(model=sbert_model)
            warmup_steps = int(len(train_dataloader) * n_epochs * 0.1) #10% of train data

            sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
                    epochs=n_epochs,
                    warmup_steps=warmup_steps)

            os.makedirs(f"{path_to_save_model}/{model_name.replace('/', '_')}")
            sbert_model.save(f"{path_to_save_model}/{model_name.replace('/', '_')}")
```
We are creating a class with 2 functions
**_prepare_training_data_** - Used to convert training data into pytorch data loader format.
**_train_sbert_** - Used to train sbert models and save them in your local directory.

This is how your training data CSV file should look like
![Training data](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/9bizehtyfk0z5oob7017.png)


```python
df = pd.read_csv('training_data.csv')
obj = trainSBERT()
obj.train_sbert(['sentence-transformers/all-MiniLM-L6-v2'], 500, df['source_sentence'].tolist(), df['target_sentence'].tolist(), "/Users/praveen/Desktop/praveen/github/training/model/sbert")
```
After 500 epochs the trained model will be saved to **/Users/praveen/Desktop/praveen/github/training/model/sbert/sentence-transformers_all-MiniLM-L6-v2**

All the below files will be saved to your local directory inside **sentence-transformers_all-MiniLM-L6-v2** folder
![Files](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/udyqwrzvwk4gdro0ceat.png)

## HOW TO USE THE TRAINED MODEL TO GENERATE EMBEDDINGS
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('/Users/praveen/Desktop/praveen/github/training/model/sbert/sentence-transformers_all-MiniLM-L6-v2')
question_embeddings = model.encode([question], convert_to_tensor=True)
answer_embeddings = model.encode([answer], convert_to_tensor=True)
print("Question Embeddings : ", question_embeddings)
print("Answer Embeddings : ", answer_embeddings)
```
Now you can compare these two using [cosine-similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to calculate how close they are.


Hope this helps :))