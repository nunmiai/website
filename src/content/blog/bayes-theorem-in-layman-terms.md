---
draft: false
title: "Bayes' Theorem In Layman Terms"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1671169151681-b33006b629ff?q=80&w=2835&auto=format&fit=crop&w=430&h=240",
    alt: "semantic search"
}
publishDate: "2023-08-28 00:00"
category: "Concepts"
author: "Praveen"
tags: [machinelearning, probability, statistics]
---

## DEFINITION
_Bayes' Theorem states that the conditional probability of an event, based on the occurrence of another event, is equal to the likelihood of the second event given the first event multiplied by the probability of the first event._

#### Blaa Blaa Blaa - I find definitions to be strange, only after understanding the concept I do understand the definition.

## Let's break it down and understand it one step at a time...

## Marginal Probability - P(A)
If a [random variable is independent](https://www.probabilitycourse.com/chapter3/3_1_4_independent_random_var.php) then the probability of the event is irrespective of the outcomes of other random variables. In simple words, it's like looking at the probability of something occurring without taking into account any other factors.

## Joint Probability - P(A,B)
The probability of 2 or more **simultaneous** events happening together. Eg Probability of watching TV and Eating.

## Conditional Probability - P(A|B)
Probability of one (or more) event given the occurrence of another event. Eg the probability of your father having dessert given that tomorrow he is having a diabetes test is very low. If you notice carefully, if there is no diabetes test tomorrow, then the probability would have been almost 100%.

## Expressing Joint Probability In Terms of Conditional Probability
$$
P(A,B) = P(A|B) \cdot P(B)
$$
Note: **P(A,B) = P(B,A)** (Symmetrical)

## Expressing Conditional Probability In Terms of Joint Probability
$$
P(A|B) = \frac{P(A,B)}{P(B)}
$$
Note: **P(A|B) \neq P(B|A)** (Not Symmetrical)

## Finally, our Bayes' Theorem using the above equations
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$
The numerator **P(B|A) \cdot P(A)** is the **joint probability** equation given above.

- P(A|B) → **Posterior Probability**
- P(B|A) → **Likelihood**
- P(A)   → **Prior Probability**
- P(B)   → **Evidence**

### Example: Probability of Fire Given Smoke
$$
P(\text{Fire}|\text{Smoke}) = \frac{P(\text{Smoke}|\text{Fire}) \cdot P(\text{Fire})}{P(\text{Smoke})}
$$
- P(Fire|Smoke) → **Posterior Probability**
- P(Smoke|Fire) → **Likelihood**
- P(Fire)       → **Prior Probability**
- P(Smoke)      → **Evidence**

The probability of fire given that there is smoke is equal to the likelihood multiplied by the probability of fire divided by the probability of smoke. This is Bayes' theorem.

## Bayes' Theorem Application in Medical Diagnosis
Consider a diagnostic test determining whether a person has a malignant lesion.

From observation, it is given that:
$$
P(\text{Test=Positive}|\text{Malignant=True}) = 0.85
$$
This means the probability of the test being positive given that the person has a malignant tumor is 85%.

Now, using Bayes' Theorem, we can calculate **P(Malignant=True | Test=Positive)**:
$$
P(\text{Malignant=True} | \text{Test=Positive}) = \frac{P(\text{Test=Positive}|\text{Malignant=True}) \cdot P(\text{Malignant=True})}{P(\text{Test=Positive})}
$$
Plugging in the known values:
$$
P(\text{Malignant=True | Test=Positive}) = \frac{0.85 \cdot 0.0002}{0.05016} = 0.003389
$$
This shows that despite a positive test, the probability of actually having a malignant tumor is only 0.33%.

## CONCLUSION
Bayes' theorem is significant in statistics and widely used in machine learning. It provides a way to update prior probabilities with new information, adjusting our beliefs based on observed data.
