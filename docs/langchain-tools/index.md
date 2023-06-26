构建 LLM 代理 (Agents) 的定制工具
====================================

Building Custom Tools for LLM Agents
---

[代理 (Agents) ](https://cookbook.langchain.com.cn/docs/langchain-agents) 是使用大型语言模型（LLM）最强大和最有趣的方法之一。LLM 的兴起使得代理 (Agents) 在基于人工智能的应用中变得非常普遍。

使用代理 (Agents) 可以让 LLM 访问工具。这些工具提供了无限的可能性。有了工具，LLM 可以搜索网络、进行数学计算、运行代码等等。

LangChain 库提供了大量预置的工具。然而，在许多真实项目中，我们通常会发现现有工具只能满足有限的需求。这意味着我们必须修改现有工具或完全构建新的工具。

本章将探讨如何在 LangChain 中为代理 (Agents) 构建定制工具。我们将从几个简单的工具开始，以帮助我们理解典型的 *工具构建模式*，然后再转向使用其他机器学习模型构建更复杂的工具，以获得更多的功能，比如描述图像。


构建工具
--------------

工具的本质是消耗一些输入（通常是字符串）并输出一些有用的信息（也通常是字符串）的对象。

实际上，它们只是我们在任何代码中都能找到的简单函数。唯一的区别是工具从 LLM 获取输入并将输出提供给 LLM。

考虑到这一点，工具相对简单。幸运的是，我们可以在很短的时间内为代理 (Agents) 构建工具。

*（请在 [此处的代码笔记本](https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/07-langchain-tools.ipynb) 中跟随）*

### 简单计算器工具

我们将从一个简单的定制工具开始。这个工具是一个简单的计算器，根据圆的半径计算圆的周长。

![根据半径计算周长](https://d33wubrfki0l68.cloudfront.net/8a6cfa9ba911326dc076ef7f898ded6e7474928c/5c65c/images/langchain-tools-3.png)


创建该工具，我们需要执行以下操作：

```python
from langchain.tools import BaseTool
from math import pi
from typing import Union
  

class CircumferenceTool(BaseTool):
      name = "Circumference calculator"
      description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)\*2.0\*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

```


在这里，我们使用 LangChain 的 `BaseTool` 对象初始化了自定义的 `CircumferenceTool` 类。我们可以将 `BaseTool` 视为 LangChain 工具的必要模板。

LangChain 要求工具具有两个属性，即 `name` 和 `description` 参数。

`description` 是工具的 *自然语言* 描述，LLM 根据它来决定是否需要使用该工具。工具描述应该非常明确，说明它们的功能、使用时机以及 *不* 适用的情况。

在我们的 `description` 中，我们没有定义不适用该工具的情况。这是因为 LLM 似乎能够识别何时需要使用此工具。在描述中添加“何时不使用”的说明对于避免工具被过度使用是有帮助的。

接下来，我们有两个方法，`_run` 和 `_arun`。当使用工具时，默认会调用 `_run` 方法。当需要 *异步* 使用工具时，会调用 `_arun` 方法。本章不涉及异步工具，所以我们用 `NotImplementedError` 对其进行了初始化。

从这里开始，我们需要为 *对话* 代理 (Agents) 初始化 LLM 和对话内存。对于 LLM，我们将使用 OpenAI 的 `gpt-3.5-turbo` 模型。要使用它，我们需要一个 [OpenAI API 密钥](https://platform.openai.com/)。

准备好后，我们可以这样初始化 LLM 和内存：

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        openai_api_key="OPENAI_API_KEY",
        temperature=0,
        model_name='gpt-3.5-turbo'
)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)

```
在这里，我们将 LLM 初始化为 `temperature` 为 `0`。在使用工具时，较低的 `temperature` 对于减少生成文本中的“随机性”或“创造性”非常有用，这对于鼓励 LLM 遵循严格的指令（如工具使用所需的指令）是理想的。

在 `conversation_memory` 对象中，我们将 `k` 设置为 `5`，以“记住”前 *五个* 人工智能交互。

现在我们可以初始化代理 (Agents) 本身了。它需要已经初始化的 `llm` 和 `conversational_memory`。它还需要一个要使用的 `tools` 列表。我们有一个工具，但我们仍然将它放入列表中。

```python
from langchain.agents import initialize_agent

tools = [CircumferenceTool()]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

```
`chat-conversation-react-description` 代理 (Agents) 类型告诉我们一些关于此代理 (Agents) 的信息，包括：
* `chat` 表示正在使用的 LLM 是一个 *聊天* 模型。`gpt-4` 和 `gpt-3.5-turbo` 都是聊天模型，因为它们消耗对话历史并生成对话响应。而像 `text-davinci-003` 这样的模型不是聊天模型，因为它不是设计成这种方式使用的。
* `conversational` 表示我们将包含 `conversation_memory`。
* `react` 指的是 [*ReAct 框架*](https://arxiv.org/abs/2210.03629)，它通过使模型能够 *“与自己对话”*，实现了多步推理和工具使用的能力。
* `description` 告诉我们 LLM/代理 (Agents) 将根据工具的描述来决定使用哪个工具——我们在之前的工具定义中创建了这些描述。

有了这一切，我们可以要求我们的代理 (Agents) 计算圆的周长。

```python
agent("can you calculate the circumference of a circle that has a radius of 7.81mm")
```
Out[]:

```python
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m{
    "action": "Final Answer",
    "action_input": "The circumference of a circle with a radius of 7.81mm is approximately 49.03mm."
}[0m
[1m> Finished chain.[0m

{'input': 'can you calculate the circumference of a circle that has a radius of 7.81mm',

 'chat_history': [],

 'output': 'The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.'}

(7.81 * 2) * pi
```
Out[]:

```python
49.071677249072565
```
代理 (Agents) 接近目标，但却不准确——出现了某些问题。我们可以在 **AgentExecutor Chain** 的输出中看到代理 (Agents) 直接跳到 **Final Answer** 操作：

```python
{ "action": "Final Answer", "action_input": "The circumference of a circle with a radius of 7.81mm is approximately 49.03mm." }

```
**Final Answer** 操作是代理 (Agents) 在决定完成推理和操作步骤并获得所有所需信息以回答用户查询时使用的操作。这意味着代理 (Agents) 决定 *不* 使用圆周计算器工具。

LLM 在数学方面通常表现不佳，但这并不能阻止它尝试进行数学计算。问题是由于 LLM 对其数学能力过于自信所致。为了解决这个问题，我们必须告诉模型它 *不能* 进行数学计算。首先，让我们看一下当前使用的提示文本：

```python
# existing prompt
print(agent.agent.llm_chain.prompt.messages[0].prompt.template)
```
Out[]:

```python
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

```
我们将添加一句话，告诉模型它在数学方面是 *“糟糕透顶的 terrible at math”*，永远不应该尝试进行数学计算。

```python
Unfortunately, the Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to its trusty tools and absolutely does NOT try to answer math questions by itself

```

将此添加到原始提示文本中后，我们使用 `agent.agent.create_prompt` 创建新的提示文本，这将为我们的代理 (Agents) 创建正确的提示结构，包括工具描述。然后，我们更新 `agent.agent.llm_chain.prompt`。


```python
sys_msg = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"""

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

agent.agent.llm_chain.prompt = new_prompt
```

现在我们可以再试一次：

```python
agent("can you calculate the circumference of a circle that has a radius of 7.81mm")
```

Out[]:

```python
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m```json
{
    "action": "Circumference calculator",
    "action_input": "7.81"
}
```[0m
Observation: [36;1m[1;3m49.071677249072565[0m
Thought:[32;1m[1;3m```json
{
    "action": "Final Answer",
    "action_input": "The circumference of a circle with a radius of 7.81mm is approximately 49.07mm."
}
```[0m
[1m> Finished chain.[0m
{'input': 'can you calculate the circumference of a circle that has a radius of 7.81mm',
 'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={})],
 'output': 'The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.'}```
```


我们可以看到代理 (Agents) 现在使用了 **Circumference calculator** 工具，并因此得到了正确的答案。

### 带有多个参数的工具

在圆周计算器中，我们只能输入一个值——`radius`——但往往我们需要多个参数。

为了演示如何实现这一点，我们将构建一个 *斜边计算器*。该工具将帮助我们计算给定三角形边长和/或角度组合的三角形斜边。

![斜边计算](https://d33wubrfki0l68.cloudfront.net/72dafa2b95666f3b390c353db46773601ef95c45/5f1dc/images/langchain-tools-4.png)

我们在这里需要多个输入，因为我们使用不同的值（边和角度）来计算三角形斜边。此外，并不需要 *所有* 值。我们可以使用任意两个或更多个参数来计算斜边。

我们这样定义新的工具：

```python
from typing import Optional
from math import sqrt, cos, sin

desc = (
    "use this tool when you need to calculate the length of a hypotenuse"
    "given one or two sides of a triangle and/or an angle (in degrees). "
    "To use the tool, you must provide at least two of the following parameters "
    "['adjacent_side', 'opposite_side', 'angle']."
)

class PythagorasTool(BaseTool):
    name = "Hypotenuse calculator"
    description = desc
    
    def _run(
        self,
        adjacent_side: Optional[Union[int, float]] = None,
        opposite_side: Optional[Union[int, float]] = None,
        angle: Optional[Union[int, float]] = None
    ):
        # check for the values we have been given
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side)\*\*2 + float(opposite_side)\*\*2)
        elif adjacent_side and angle:
            return adjacent_side / cos(float(angle))
        elif opposite_side and angle:
            return opposite_side / sin(float(angle))
        else:
            return "Could not calculate the hypotenuse of the triangle. Need two or more of `adjacent_side`, `opposite_side`, or `angle`."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

tools = [PythagorasTool()]

```
在工具描述中，我们用自然语言描述了工具的功能，并指定了“要使用该工具，必须提供以下至少两个参数 [‘adjacent_side’，‘opposite_side’，‘angle’]”的说明。这个指导是我们为了让 `gpt-3.5-turbo` 了解函数所需的输入格式所需要的。

与之前一样，我们必须更新代理 (Agents) 的提示。我们不需要修改系统消息，但是我们需要更新提示中描述的可用工具。


再次，我们看到正确的工具使用。即使在我们简短的工具描述中，代理 (Agents) 也能始终按照预期和使用多个参数使用该工具。

### 更高级的工具使用
我们已经看到了两个自定义工具的示例。在大多数情况下，我们可能希望做一些更强大的事情-所以让我们试试看。

受 HuggingGPT 论文 [1] 的启发，我们将采用一个现有的开源 *专家模型*，该模型已经针对我们的 LLM 无法执行的特定任务进行了训练。

该模型是 [`Salesforce/blip-image-captioning-large`](https://huggingface.co/Salesforce/blip-image-captioning-large)，托管在 Hugging Face 上。该模型接收一张图片并对其进行描述，这是我们的 LLM 无法做到的。
首先，我们需要这样初始化模型：

我们将按照以下步骤进行操作：
1. 下载一张图片。
2. 将其作为 Python PIL 对象（图像数据类型）打开。
3. 使用 `processor` 调整图片大小和归一化。
4. 使用 `model` 生成描述。

让我们从第一步和第二步开始：



虽然猩猩 *技术上* 不是猴子，但这个描述还是相当准确的。我们的代码有效。现在让我们将这些步骤整合成一个我们的代理 (Agents) 可以使用的工具。


我们重新初始化代理 (Agents) 提示（删除现在不再需要的 *“你不能进行数学运算”* 指令），并设置 `tools` 属性以反映新的工具列表：


现在我们可以继续要求代理 (Agents) 描述上述相同的图片，将其 URL 传递给查询。

```python
agent(f"What does this image show?\n{img_url}")
```
Out[]:
```python
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m{
    "action": "Image captioner",
    "action_input": "https://images.unsplash.com/photo-1616128417859-3a984dd35f02?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2372&q=80"
}[0m


Observation: [36;1m[1;3mthere is a monkey that is sitting in a tree[0m


Thought:[32;1m[1;3m{


    "action": "Final Answer",


    "action_input": "There is a monkey that is sitting in a tree."


}[0m

[1m> Finished chain.[0m


{'input': 'What does this image show?\nhttps://images.unsplash.com/photo-1616128417859-3a984dd35f02?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2372&q=80',


 'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),


  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={}),


  HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),


  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.', additional_kwargs={}),


  HumanMessage(content='If I have a triangle with two sides of length 51cm and 34cm, what is the length of the hypotenuse?', additional_kwargs={}),


  AIMessage(content='The length of the hypotenuse is approximately 61.29cm.', additional_kwargs={})],


 'output': 'There is a monkey that is sitting in a tree.'}
```
让我们再试一些：
![冲浪的人](https://d33wubrfki0l68.cloudfront.net/1f3f639410a3f137375a36543f4a35d59c519c10/dd757/images/langchain-tools-1.png)

```python
img_url = "https://images.unsplash.com/photo-1502680390469-be75c86b636f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80"
agent(f"what is in this image?\n{img_url}")
```
Out[]:
```python
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m{
    "action": "Image captioner",
    "action_input": "https://images.unsplash.com/photo-1502680390469-be75c86b636f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80"
}[0m


Observation: [36;1m[1;3msurfer riding a wave in the ocean on a clear day[0m


Thought:[32;1m[1;3m{
    "action": "Final Answer",
    "action_input": "The image shows a surfer riding a wave in the ocean on a clear day."
}[0m
[1m> Finished chain.[0m
{'input': 'what is in this image?\nhttps://images.unsplash.com/photo-1502680390469-be75c86b636f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80',
 'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={}),
  HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.', additional_kwargs={}),
  HumanMessage(content='If I have a triangle with two sides of length 51cm and 34cm, what is the length of the hypotenuse?', additional_kwargs={}),
  AIMessage(content='The length of the hypotenuse is approximately 61.29cm.', additional_kwargs={}),
  HumanMessage(content='What does this image show?\nhttps://images.unsplash.com/photo-1616128417859-3a984dd35f02?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2372&q=80', additional_kwargs={}),
  AIMessage(content='There is a monkey that is sitting in a tree.', additional_kwargs={})],
 'output': 'The image shows a surfer riding a wave in the ocean on a clear day.'}
```
这是另一个准确的描述。让我们尝试一些更具挑战性的东西：
![小鳄鱼站在一根木头上](https://d33wubrfki0l68.cloudfront.net/e4814f6d30b87543ed4a722ae4e8199fa3b58624/8a74f/images/langchain-tools-2.png)

```python
img_url = "https://images.unsplash.com/photo-1680382948929-2d092cd01263?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2365&q=80"
agent(f"what is in this image?\n{img_url}")
```
Out[]:
```python
[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m```json
{
    "action": "Image captioner",
    "action_input": "https://images.unsplash.com/photo-1680382948929-2d092cd01263?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2365&q=80"
}
```[0m
Observation: [36;1m[1;3mthere is a lizard that is sitting on a tree branch in the water[0m
Thought:[32;1m[1;3m```json
{
    "action": "Final Answer",
    "action_input": "There is a lizard that is sitting on a tree branch in the water."
}
```[0m
[1m> Finished chain.[0m
{'input': 'what is in this image?\nhttps://images.unsplash.com/photo-1680382948929-2d092cd01263?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2365&q=80',
 'chat_history': [HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.03mm.', additional_kwargs={}),
  HumanMessage(content='can you calculate the circumference of a circle that has a radius of 7.81mm', additional_kwargs={}),
  AIMessage(content='The circumference of a circle with a radius of 7.81mm is approximately 49.07mm.', additional_kwargs={}),
  HumanMessage(content='If I have a triangle with two sides of length 51cm and 34cm, what is the length of the hypotenuse?', additional_kwargs={}),
  AIMessage(content='The length of the hypotenuse is approximately 61.29cm.', additional_kwargs={}),
  HumanMessage(content='What does this image show?\nhttps://images.unsplash.com/photo-1616128417859-3a984dd35f02?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2372&q=80', additional_kwargs={}),
  AIMessage(content='There is a monkey that is sitting in a tree.', additional_kwargs={}),
  HumanMessage(content='what is in this image?\nhttps://images.unsplash.com/photo-1502680390469-be75c86b636f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80', additional_kwargs={}),
  AIMessage(content='The image shows a surfer riding a wave in the ocean on a clear day.', additional_kwargs={})],
 'output': 'There is a lizard that is sitting on a tree branch in the water.'}
```
与 *鳄鱼* 而不是 *蜥蜴* 稍有不准确，但除此之外，标题很好。


---

我们已经探索了如何为 LangChain 代理 (Agents) 构建自定义工具。这个功能极大地扩展了大型语言模型的可能性。

在我们简单的例子中，我们看到了 LangChain 工具的典型结构，然后继续将 *专家模型* 作为工具添加进来，其中我们的代理 (Agents) 作为这些模型的 *控制器*。

当然，我们可以做的远不止我们在这里展示的内容。工具可以用于与无尽的功能和服务集成，或者与一系列专家模型进行通信，就像 HuggingGPT 所展示的那样。

我们通常可以使用 LangChain 的默认工具来运行 SQL 查询，执行计算或进行向量搜索。但是，当这些默认工具无法满足我们的要求时，我们现在知道如何构建自己的工具。


---
参考资料
----------
[1] Y. Shen, K. Song, et al., [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://arxiv.org/abs/2303.17580) (2023)

