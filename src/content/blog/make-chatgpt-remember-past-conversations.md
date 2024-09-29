---
draft: false
title: "Make ChatGPT keep track of your past conversations"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?q=80&w=2940&auto=format&fit=crop&w=430&h=240",
    alt: "semantic search"
}
publishDate: "2023-05-11 00:00"
category: "Tutorials"
author: "Adheeban"
tags: [Python, GenAI, Langchain]
---

In our previous blog, we fed ChatGPT with the custom knowledge from documents and made it answer our queries using Llama-Index. One problem with the previous implementation is that the chatbot wouldn't be able to answer any of our questions addressing the previous questions/prompts. Here's what I'm talking about:

```
ASK: You are an AI assistant, specialized in programming.                                    

----------------------------------------

ChatGPT says: 

Hello! I'm here to assist you with anything related to
programming. What can I help you with today? 

########################################

ASK: what did I tell you earlier?                                  

----------------------------------------

ChatGPT says: 

I'm sorry, but as an AI language model, I don't have access to
any context or information about our previous interactions
unless you provide it to me. Could you please clarify what you
asked me earlier? I'll do my best to assist you.
```

So, lets set the custom knowledge base aside for a short-while and focus on this memory problem in this blog. We will be using the Langchain framework that offers various kinds of memory classes that will help us to build a chatbot system that keeps track of the past conversations we had with it.

If you ever visited [chat.openai.com](chat.openai.com), you would know that OpenAI maintains histories of chats, ChatGPT had with the user in seperate containers called sessions. Basically, we'll be trying to mimic these sessions using langchain. 

Now, that should give you an idea. Before we start, here are some more things you should know.

- OpenAI's ChatGPT has a token limit of 4096 (1 token =~ 4 characters)
- This limit includes both your prompt and the response (completion tokens) that is returned from ChatGPT
- Anything that has to be generated post this token limit will be ignored abruptly without even a single warning
- So, ideally we should be keeping our prompt token limit with 2048 (half the limit) and leave the rest for ChatGPT's completion. This buffer changes based on your use case.

Well, why am I blabbering all this now? I'll explain in a while. Now, without further ado, let's get started with langchain.

---

## Enter Langchain :parrot::link:

So, what's our goal? We don't have to worry about feeding document context into the LLM for now. We rather have to build a simple chatbot that remembers our past conversations. With langchain, it is as easy as 1, 2, 3 !


```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_KEY"] = 'Your API Key Here'

llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512)

conversation = ConversationChain(
    llm=llm, 
    # verbose=True, 
    memory=ConversationBufferMemory()
)

def chatbot(pt):
    res = conversation.predict(input=pt)
    return res

if __name__=='__main__':
    while True:
        print('########################################\n')
        pt = input('ASK: ')
        if pt.lower()=='end':
            break
        response = chatbot(pt)
        print('\n----------------------------------------\n')
        print('ChatGPT says: \n')
        print(response, '\n')
```

That's all you gotta do. Now you know the drill, copy the above snippet and name the file whatever you want and run it with `python3 <filename>.py>`

Here's the conversation I had:

```
ASK: Hello, who are you?

----------------------------------------

ChatGPT says: 

Hello! I am an artificial intelligence program designed to
interact with humans and assist with various tasks. How can I
help you today? 

########################################

ASK: Who was martin luther king?

----------------------------------------

ChatGPT says: 

Martin Luther King Jr. was an American Baptist minister and
activist who became the most visible spokesperson and leader
in the civil rights movement from 1954 until his assassination
in 1968.

########################################

ASK: what was his profession?

----------------------------------------

ChatGPT says: 

Martin Luther King Jr. was a Baptist minister and activist. He
became a prominent leader in the civil rights movement and
advocated for nonviolent civil disobedience to advance civil
rights for African Americans. 

########################################

ASK: Okay, what was my first question to you?

----------------------------------------

ChatGPT says: 

Your first question to me was "Hello, who are you?" 
```

As you could clearly see, now the LLM is able to remember my past conversations. Even the first question I asked. In the code setting the `verbose` argument to `True` during the initialization of `ConversationChain` class is upto you. Setting the option to true will give you a glimpse of what's happening in the backend as we sail through the conversation with the LLM.

Our conversations will be stored in the Memory class of langchain, if you want to store the conversation for later retrieval, you can do so by pickling the `chain.messages` class and save it in a file. Later when you need to load your history, you just have to overwrite your chain's `messages` class with the pickled string, like how I have done below:

```python
qa = ConversationChain(
            llm=llm, memory=memory)
        
with open(f'sessions/{self.history_key}','rb') as f:
     user_ses = pickle.loads(f.read())
     qa.memory = user_ses
```

--- 
## Enough with the code! Now, What's happening underneath?
In the example code snippet above, I have used the `ConversationBufferMemory` memory class, this is the most simplest implementation of memory in langchain. There are other memory classes too in langchain, each with its own perk. Before I get into that, Here's what happens when you have a conversation with a chain with memory in langchain:

- When you send in the first question or prompt to the LLM, it returns a response. Cool.
- Now, when you send the second prompt, langchain also sends the previous conversation the user had with the LLM as context for the LLM, so that the LLM would be able to answer any questions addressing the previous questions.
- This happens with every conversation. Previous conversations will be linked as context to the LLM with every hit to the LLM's API.

A very simple engineering problem handled effortlessly with langchain. Now, Remember when I told you about the token limit in ChatGPT? It plays a crucial role now. Whatever conversation you had previously with the LLM will be linked with your original prompt and sent to the API. As the depth of the conversation increases, the percentage of tokens the history uses in the whole 4096 token limit increases rapidly. Which will eventually leave no room for the LLM's response to the current question. 

Langchain knows about this problem very well, that's why they have different memory classes in their memory module that suits your unique use case, I'll give short note on a impressive few.

`ConversationSummaryMemory` - This type of memory creates a summary of the conversation over time. This can be useful for condensing information from the conversation over time resulting in a lesser usage of tokens within the 4096 limit. 

`ConversationKGMemory` - This type of memory uses a knowledge graph to recreate memory. It identifies the entities and maps out the relation between each of those entities.

These memories use the LLM itself to identify entities, to summarize the previous conversations etc. This should be taken into account as this is an extra hit to the API before every conversation. There are a lot more memory classes in langchain, Be sure you give langchain's documentation a read [here](https://python.langchain.com/en/latest/modules/memory/how_to_guides.html)

---

Now, that's about Langchain's memory classes. In the next blog of this series, we'll try to combine the vectore store from the first blog with the memory classes in this blog. LlamaIndex + Langchain :fire:. See ya in the next one :wink:
