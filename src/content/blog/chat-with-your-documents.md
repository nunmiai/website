---
draft: false
title: "Chat with your documents using ChatGPT"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1593349480506-8433634cdcbe?q=80&w=2940&auto=format&fit=crop&w=430&h=240",
    alt: "RAG"
}
publishDate: "2023-05-06 11:39"
category: "Tutorials"
author: "Adheeban"
tags: [ChatGPT, OpenAI, RAG]
---

Ever since [OpenAI](https://openai.com/) announced their language model [ChatGPT](https://openai.com/blog/chatgpt), it has been making headlines in the AI world on a daily basis. ChatGPT is being used as the foundation for countless new tools and applications, ranging from customer service chatbots to creative writing assistants. With its ability to generate high-quality, human-like responses to complex prompts, ChatGPT has quickly become a game-changing technology. 

The applications of LLMs like ChatGPT are virtually limitless. Our imagination would be the only barrier. In this blog series though, We'll be focusing on how we can make ChatGPT, or any LLM for that matter to answer our queries with the context of the custom knowledge from the documents we feed to it. We'll start off with a simpler implementation using [Llama-index](https://github.com/jerryjliu/llama_index) that reads almost all types of basic document formats and returns the response to your queries based upon it. As we progress, we'll be using [Langchain](https://python.langchain.com/en/latest/index.html) to build a full fledged chatbot framework that reads the content from almost any link or document that you give to it and answers your queries accordingly. Langchain is a great framework for developing applications powered by language models. We'll be building a web app using these frameworks. Exciting times ahead !

### Let me lay the **prerequisites** first:
- You will need a working OpenAI key, because we'll be using the GPT-3 model underneath. If you don't have one, here's [how to get an OpenAI API Key](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/)
- You need to have python>=3.6 installed in your machine

That's about everything you'd need. Rest of the things we'll take care of, as we sail through. Now without further ado, let's dive right in.

---

## Behold, the power of Llama-Index :llama:

As I said earlier, We are gonna use llama_index for this tutorial. We are not building anything fancy as of now. We'll not be building any UI. The sole purpose of this is to give an understanding of how llama_index works underneath. Below is the implementation using llama_index and langchain. llama_index uses langchain models under the hood.
```python
from gpt_index import download_loader, SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = 'Your API Key Here'

file_path = input('Enter the path of the file/doc: ')

def build_index(file_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 256

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    download_loader('SimpleDirectoryReader')
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return index


index = build_index(file_path=file_path)

def chatbot(prompt):
    return index.query(prompt, response_mode="compact")
   
while True:
    print('########################################')
    pt = input('ASK: ')
    if pt.lower()=='end':
        break
    response = chatbot(pt)
    print('----------------------------------------')
    print('ChatGPT says: ')
    print(response)
```

Copy the above code in entirety and paste it in a file and name it whatever you want. I'm naming it `main.py`. Replace the API key placeholder in the code, with your own OpenAI API key and you are done.

First up, run the below command to install the libraries that we'd need:

```
pip3 install gpt_index==0.4.24 && pip3 install langchain==0.0.142
```

You can run the code by running this command `python3 main.py` at the location of this file, when prompted give the path of your context file to the program. I'm using Martin Luther King's "[I have a dream](https://www.btboces.org/Downloads/I%20Have%20a%20Dream%20by%20Martin%20Luther%20King%20Jr.pdf)" speech transcript PDF file.

```bash
➜  python3 main.py
Enter the path of the file/doc: blogs/docbot/martin.pdf
```

You can give any type of file format. Depending on the file
size and your machine's power, it'll take some time to read and convert your file contents into vectors. In my case, it takes around 10 seconds to process a 30MB document. 

Once it is done, you can start asking questions to your document using ChatGPT. Below are some of the example prompts and responses.

```
ASK: 

what is this document about?

----------------------------------------

ChatGPT says: 

The document is about a speech or written piece that discusses
the promise of equal rights for all Americans, particularly
black Americans, as outlined in the Constitution and Declaration of Independence.
It addresses the fact that this promise has not been fully
realized and that there is a shameful condition that needs to be dramatized.  
```

```
ASK: 

summarize the speech

----------------------------------------

ChatGPT says: 

The speech urges America to remember the importance of freedom and justice, and emphasizes the urgent
need for action in the face of racial injustice. The speaker encourages the audience to rise above 
segregation and work towards brotherhood. The overall message is that now is the time to make real the
promises of democracy. 
```

That's it. As simple as that. 35 lines of code. Llama-Index and other tools like it have made it so easy and user friendly to leverage the full power of LLMs.

---

## Okay that's cool, but what really happens underneath?

Llama-index and langchain have made the whole process very seamless, which otherwise would've been a really cumbersome task. So, here's what happens:

- When you give llama_index the document, it uses one of the adapters that actually suits for the file type from the collection of pre-built adapters from the [Llama-Hub](https://llamahub.ai/) and parses the contents of the file.

- Once the parsing is done, llama_index converts the whole content into chunks of vectors. 

- When you put up a question to ChatGPT, llama_index takes in your question retrieves the chunks of vectors from the parsed file that are relevant to your prompt using [similarity search](https://www.pinecone.io/learn/what-is-similarity-search/)

- Once it retrieves the relevant chunks, llama_index overrides your original prompt by adding what it retrieved as the context to the model. 

- With the original question and the context it just has been provided with, ChatGPT should be able to understand your question.

And voila! you'll get a relevant response from ChatGPT. This is how llama_index makes LLMs understand custom knowledge. There's more to this and new features are getting added to llama_index everyday. Make sure you explore more of what it could do.

---
Now, that's about llama_index. In the next blog of this series. We'll see how to use langchain to build a more robust chatbot framework that keeps track of the previous conversations it had with the user. See ya in the next one :wink:
