---
draft: false
title: "Tools and Tool Choice with Azure"
snippet: "Ornare cum cursus laoreet sagittis nunc fusce posuere per euismod dis vehicula a, semper fames lacus maecenas dictumst pulvinar neque enim non potenti. Torquent hac sociosqu eleifend potenti."
image: {
    src: "https://images.unsplash.com/photo-1509803874385-db7c23652552?q=80&w=2940&auto=format&fit=crop&w=430&h=240",
    alt: "semantic search"
}
publishDate: "2024-06-13 00:00"
category: "Tutorials"
author: "Praveen"
tags: [machinelearning, probability, statistics]
---

When it comes to integrating GPT into our products and especially if a chain of logical decisions are made based on GPT's result then we have to worry about the unstructured nature of GPT's response.

There are several ways to solve this issue

1. Prompt Engineering - Emphasizing to return a structured output maybe as a JSON. This technique might work but sometimes the result could still be unstructured.
2. **Langchain**, **Llamaindex** and **DSPy** offer several functionalities to generate structured output and these techniques are usually robust but not native.

In this blog we are going to see a native way to get structured output from GPT4 and the granularity of control is that each of the returned parameter's data type could even be specified and obtained.

## Let's look at an example ...
We are going to ask GPT4 few problems to solve and the expected result should have

1. **formula** - formula used to solve the problem
2. **substitution** - substitute the values from the problem in the formula
3. **result** - final answer post substitution
4. **explanation** - a simple explanation on what the problem is and how to solve it
5. **difficulty** - on a scale of 1-10 how difficult is this problem for an engineering student

So if we are not going to use any frameworks, few shot examples with properly defined output structure in prompt might help us get output in the desired format but natively we have something called **tools** and **tool_choice** to make the output structured. These features were released so that the output of GPT could be obtained as **parameters** and these parameters could be used to **call a function**.


![Flow](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/p559ctx3w8vpwfw8alf4.png)

## Let's look at some code
### Installation
```
pip install openai
```

### Defining the output structure that we want
```python
tools = [
        {
            "type": "function",
            "function": {
                "name": "problem_solver",
                "description": "Used to solve the problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula": {
                            "type": "string",
                            "description": "formula used to solve the problem",
                        },
                        "substitution": {
                            "type": "string",
                            "description": "substitute the values present in the problem into the formula used to solve the problem",
                        },
                        "result": {
                            "type": "string",
                            "description": "the final answer for the problem in float",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "explanation on how the problem is solved in simple words",
                        },
                        "difficulty": {
                            "type": "integer",
                            "description": "on a scale of 1-10, how difficult is the problem to solve for an enginnering student",
                        },
                    },
                    "required": ["formula", "substitution", "result", "explanation", "difficulty"],
                },
            },
        }
    ]
```

- Name of the function is **problem_solver**

- The 5 parameters that we want in the output are **formula**, **substitution**, **result**, **explanation**, **difficulty** which are defined inside **parameters** ==> **properties**.

- For each of the above parameters the **type**, **description** should be defined which specifies the **data type** of the returned value and **simple description** of what is to be returned.

- In the **required** key which is a list we have to specify the mandatory parameters that have to be returned otherwise GPT might consider it to be optional and might not return it.

### Defining a function which is used to call Azure GPT model
```python
def solve_problem(messages_list):
    api_key = 'your_api_key'
    api_base = 'your_api_base_url'
    api_version = '2024-02-01'
    model = 'your_deployment_name'
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key=api_key,
        api_version=api_version
        )
    response = client.chat.completions.create(
            model=model,
            temperature=0.8,
            max_tokens=500,
            messages=messages_list,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "problem_solver"}}
        )
    res=response.choices[0].message
    return res
```

- In the **client.chat.completions.create** we have two parameters **tools** and **tool_choice**, for the tools parameter we can pass the tools list we created before.

- The **tool_choice** parameter accepts three values

1. 'auto' - This is the default value when we define parameters in the above step and pass it to tools. By specifying 'auto', we allow GPT to choose the function and parameters that we have defined in tools, sometimes GPT might not choose our defined function and parameters so there is a bit of uncertainty with 'auto'.
2. None - This is the default value when no function and parameters are defined. This is a way to specify not to use this feature.
3. Specifying a particular function via **{"type: "function", "function": {"name": "my_function"}}** forces the model to call that function. In our case this would be **{"type: "function", "function": {"name": "problem_solver"}}**. This **reinforces** GPT to return the parameters defined under problem_solver.

### Let's try asking few questions
```python
result = solve_problem([{
    "role":"user",
    "content": "An airplane accelerates down a runway at 3.20 m/s2 for 32.8 s until is finally lifts off the ground. Determine the distance traveled before takeoff"
}])
tool_calls = result.tool_calls
parameters = eval(tool_calls[0].function.arguments)
print(parameters)
```

### OUTPUT
```
{'formula': 'd = v_i * t + (1/2) * a * t^2',
 'substitution': 'd = 0 * 32.8 + (1/2) * 3.20 * (32.8)^2',
 'result': '1721.472 m',
 'explanation': 'Since the airplane starts from rest, its initial velocity (v_i) is 0. The acceleration (a) is 3.20 m/s2 and the time (t) is 32.8 seconds. Using the kinematic equation for distance (d), where the first term is zero because the initial velocity is zero, the second term is (1/2) * acceleration * time squared gives the distance. After calculating, the distance comes out to be 1721.472 meters.',
 'difficulty': 3}
```
We can observe that the output is now a proper python dictionary which is easily parsable and could be used to call any function.
Any if you want any more customization in the parameter data types refer to [https://json-schema.org/understanding-json-schema/reference/type](https://json-schema.org/understanding-json-schema/reference/type).

We can define multiple functions and parameters in tools and let GPT decide which function and parameter to use based on prompt and description provided using 'auto' as tool_choice value or could enforce use of a particular function and its parameters by specifying it in tool_choice.

Hope this helps :))
