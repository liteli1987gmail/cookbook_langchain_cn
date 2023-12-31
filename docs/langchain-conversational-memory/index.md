使用 Langchain 的 LLM 的对话记忆 
=============================================

对话记忆是指聊天机器人可以以聊天的方式对多个查询进行响应。它使得对话连贯，并且如果没有它，每个查询都将被视为完全独立的输入，而不考虑过去的交互。

![具有和不具有对话记忆的情况](https://d33wubrfki0l68.cloudfront.net/41a8697fd44e325cfea1537aa04e1b8ab8cd0f94/915dd/images/langchain-conversational-memory-1.png '具有和不具有对话记忆的 LLM') 


蓝色框是用户提示，灰色是 LLM 的响应。左侧把聊天记录存储起来了，最后 LLM 回答出来了。但是没有对话记忆（右侧），LLM 无法使用先前交互的知识进行响应。

记忆 ( memory )允许大型语言模型（LLM）记住与用户的先前交互。

默认情况下，LLM 是 `无状态 stateless` 的，这意味着每个传入的查询都独立处理，不考虑其他交互。对于无状态代理 (Agents) 来说，唯一存在的是当前输入，没有其他内容。

有许多应用场景，记住先前的交互非常重要，比如聊天机器人。 ` ConversationChain ` 对话记忆使我们能够做到这一点。

有几种方法可以实现对话记忆。在 [LangChain](https://cookbook.langchain.com.cn/docs/langchain-intro/) 的上下文中，它们都是构建在 ` ConversationChain ` 之上的。

ConversationChain
-----------------
我们可以通过初始化 ` ConversationChain ` 来开始。我们将使用 OpenAI 的 ` text-davinci-003 ` 作为 LLM，但也可以使用其他模型，比如 `gpt-3.5-turbo` 。

```python
from langchain import OpenAI
from langchain.chains import ConversationChain

# first initialize the large language model
llm = OpenAI(
	temperature=0,
	openai_api_key="OPENAI_API_KEY",
	model_name="text-davinci-003"
)

# now initialize the conversation chain
conversation = ConversationChain(llm=llm)

```
我们可以这样查看 `ConversationChain` 的提示模板：

In [8]:
```python
print(conversation.prompt.template)
```
Out[8]:
```python
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

{history}

Human: {input}

AI:

```
在这里，提示模板告诉模型以下内容是人类 Human（我们）与 AI（`text-davinci-003`）之间的对话。通过陈述这些内容，提示模板试图减少*幻觉*（模型编造事物）的发生：

```python
"If the AI does not know the answer to a question, it truthfully says it does not know."

```
这可以帮助，但不能解决幻觉问题，但我们将把这个问题留给未来的章节讨论。

在初始提示之后，我们看到了两个参数： `{history}` 和 `{input}` 。 `{input}` 是我们将放置最新的人类查询的地方；它是输入到聊天机器人文本框中的内容：

![ChatGPT对话截图显示聊天历史和输入](https://d33wubrfki0l68.cloudfront.net/969aa908ca7190f698df42b4230e3d4594c6fb76/48a9f/images/langchain-conversational-memory-2.png)

 `{history}` 是使用对话记忆的地方。在这里，我们提供有关人类和AI之间对话历史的信息。

这两个参数—— `{history}` 和 `{input}` ——被传递到刚刚看到的提示模板中的LLM中，我们（希望）返回的输出只是对话的预测延续。

对话记忆的形式 Forms of Conversational Memory
------------------------------
我们可以使用多种类型的对话记忆来使用 `ConversationChain` 。它们会修改传递给 `{history}` 参数的文本。

### ConversationBufferMemory

*(请参考我们的[Jupyter笔记本](https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/03-langchain-conversational-memory.ipynb))*

`ConversationBufferMemory`是LangChain 中最直接的对话记忆形式。

如上所述，过去人类和AI之间的原始对话输入以其原始形式传递给 `{history}` 参数。

In[11]:
```python
from langchain.chains.conversation.memory import ConversationBufferMemory

conversation_buf = ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory()
)
```
In[32]:
```python
conversation_buf("Good morning AI!")
```
Out[32]:
```python
{'input': 'Good morning AI!',
 'history': '',
 'response': " Good morning! It's a beautiful day today, isn't it? How can I help you?"}
```
我们返回了对话代理 (Agents) 的第一个响应。

让我们继续对话，编写只有在 LLM 考虑对话历史时才能回答的提示。

我们还添加了一个`count_tokens`函数，以便我们可以看到每个交互使用了多少令牌 （Tokens） 。

In[6]:
```python
from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result
```
In[33]:
```python
count_tokens(
    conversation_buf, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)
```

Out[33]:
```python
Spent a total of 179 tokens
"""Interesting! Large Language Models are a type of artificial intelligence that can process natural language and generate text.
 They can be used to generate text from a given context, or to answer questions about a given context. 
 Integrating them with external knowledge can help them to better understand the context and generate more accurate results. 
 Is there anything else I can help you with?"""
```
In[34]:
```python
count_tokens(
    conversation_buf,
    "I just want to analyze the different possibilities. What can you think of?"
)
```
Out[34]:
```python
Spent a total of 268 tokens
' Well, integrating Large Language Models with external knowledge can open up a lot of possibilities. For example, you could use them to generate more accurate and detailed summaries of text, or to answer questions about a given context more accurately. You could also use them to generate more accurate translations, or to generate more accurate predictions about future events.'
```
In[35]:
```python
count_tokens(
    conversation_buf, 
    "Which data source types could be used to give context to the model?"
)
```
Out[35]:
```python
Spent a total of 360 tokens
'  There are a variety of data sources that could be used to give context to a Large Language Model. These include structured data sources such as databases, unstructured data sources such as text documents, and even audio and video data sources. Additionally, you could use external knowledge sources such as Wikipedia or other online encyclopedias to provide additional context.'
```
In[36]:
```python
count_tokens(
    conversation_buf, 
    "What is my aim again?"
)
```
Out[36]:
```python
Spent a total of 388 tokens
' Your aim is to explore the potential of integrating Large Language Models with external knowledge.'
```
LLM可以清楚地记住对话的历史。让我们来看一下`ConversationBufferMemory`是如何存储对话历史的：
In[37]:
```python
print(conversation_buf.memory.buffer)
```
Out[37]:
```python

Human: Good morning AI!
AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
Human: My interest here is to explore the potential of integrating Large Language Models with external knowledge
AI:  Interesting! Large Language Models are a type of artificial intelligence that can process natural language and generate text. They can be used to generate text from a given context, or to answer questions about a given context. Integrating them with external knowledge can help them to better understand the context and generate more accurate results. Is there anything else I can help you with?
Human: I just want to analyze the different possibilities. What can you think of?
AI:  Well, integrating Large Language Models with external knowledge can open up a lot of possibilities. For example, you could use them to generate more accurate and detailed summaries of text, or to answer questions about a given context more accurately. You could also use them to generate more accurate translations, or to generate more accurate predictions about future events.
Human: Which data source types could be used to give context to the model?
AI:   There are a variety of data sources that could be used to give context to a Large Language Model. These include structured data sources such as databases, unstructured data sources such as text documents, and even audio and video data sources. Additionally, you could use external knowledge sources such as Wikipedia or other online encyclopedias to provide additional context.
Human: What is my aim again?
AI:  Your aim is to explore the potential of integrating Large Language Models with external knowledge.

```
我们可以看到缓冲区直接保存了聊天历史中的每个交互。这种方法有一些优缺点。

简而言之，它们是：


| 优点 | 缺点 |
| --- | --- |
| 存储所有内容为LLM提供了最大数量的信息 | 更多令牌 （Tokens） 意味着响应时间更长和更高的成本 |
| 存储所有内容简单直观 | 较长的对话无法记住，因为我们达到了LLM的令牌 （Tokens） 限制（`text-davinci-003`和`gpt-3.5-turbo`的令牌 （Tokens） 限制为`4096`） |

`ConversationBufferMemory`是一个很好的选择，但受到存储每个交互的限制。

让我们来看一下其他有助于解决这个问题的选项。

### ConversationSummaryMemory

使用`ConversationBufferMemory`，我们很快就会使用*大量*的令牌 （Tokens） ，甚至超过了当今最先进的LLM的上下文窗口限制。
为了避免过多使用令牌 （Tokens） ，我们可以使用`ConversationSummaryMemory`。

顾名思义，这种记忆形式在传递给 `{history}` 参数之前对对话历史进行*总结*。

我们可以这样初始化`ConversationChain`以使用总结记忆：

```python
from langchain.chains.conversation.memory import ConversationSummaryMemory

conversation = ConversationChain(
	llm = llm,
	memory = ConversationSummaryMemory(llm = llm)
)

```
当使用`ConversationSummaryMemory`时，我们需要向对象传递一个LLM，因为总结是由LLM提供支持的。

我们可以在这里看到用于此操作的提示：

In[19]:
```python
print(conversation_sum.memory.prompt.template)
```
Out[19]:
```python
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:

```
使用这种方法，我们可以总结每个新的交互，并将其附加到所有过去交互的 `运行汇总` 中。

让我们再次进行一次对话，利用这种方法。

In[40]:
```python
# without count_tokens we'd call `conversation_sum("Good morning AI!")`
# but let's keep track of our tokens:
count_tokens(
    conversation_sum, 
    "Good morning AI!"
)
```
Out[40]:
```python
Spent a total of 290 tokens
" Good morning! It's a beautiful day today, isn't it? How can I help you?"
```
In[41]:
```python
count_tokens(
    conversation_sum, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)
```
Out[41]:
```python
Spent a total of 440 tokens
" That sounds like an interesting project! I'm familiar with Large Language Models, but I'm not sure how they could be integrated with external knowledge. Could you tell me more about what you have in mind?"
```
In[42]:
```python
count_tokens(
    conversation_sum, 
    "I just want to analyze the different possibilities. What can you think of?"
)
```
Out[42]:
```python
Spent a total of 664 tokens
' I can think of a few possibilities. One option is to use a large language model to generate a set of candidate answers to a given query, and then use external knowledge to filter out the most relevant answers. Another option is to use the large language model to generate a set of candidate answers, and then use external knowledge to score and rank the answers. Finally, you could use the large language model to generate a set of candidate answers, and then use external knowledge to refine the answers.'
```
In[43]:
```python
count_tokens(
    conversation_sum, 
    "Which data source types could be used to give context to the model?"
)
```
Out[43]:
```python
Spent a total of 799 tokens
' There are many different types of data sources that could be used to give context to the model. These could include structured data sources such as databases, unstructured data sources such as text documents, or even external APIs that provide access to external knowledge. Additionally, the model could be trained on a combination of these data sources to provide a more comprehensive understanding of the context.'
```
In[44]:
```python
count_tokens(
    conversation_sum, 
    "What is my aim again?"
)
```
Out[44]:
```python
Spent a total of 853 tokens
' Your aim is to explore the potential of integrating Large Language Models with external knowledge.'
```
在这种情况下，总结包含了足够的信息，以便 LLM *"记住"* 我们最初的目标。

我们可以这样看到这个总结的原始形式：

In[45]:
```python
print(conversation_sum.memory.buffer)
```
Out[45]:
```python

The human greeted the AI with a good morning, to which the AI responded with a good morning and asked how it could help. The human expressed interest in exploring the potential of integrating Large Language Models with external knowledge, to which the AI responded positively and asked for more information. The human asked the AI to think of different possibilities, and the AI suggested three options: using the large language model to generate a set of candidate answers and then using external knowledge to filter out the most relevant answers, score and rank the answers, or refine the answers. The human then asked which data source types could be used to give context to the model, to which the AI responded that there are many different types of data sources that could be used, such as structured data sources, unstructured data sources, or external APIs. Additionally, the model could be trained on a combination of these data sources to provide a more comprehensive understanding of the context. The human then asked what their aim was again, to which the AI responded that their aim was to explore the potential of integrating Large Language Models with external knowledge.

```

与使用`ConversationBufferMemory`相比，此对话使用了更多的令牌 （Tokens） ，那么使用`ConversationSummaryMemory`是否有任何优势呢？

![随着交互数量增加，令牌 （Tokens） 数量的变化](https://d33wubrfki0l68.cloudfront.net/88c3a63dd5bdf04ad00da7f6b3a192f2340b271a/058e3/images/langchain-conversational-memory-3.png)

令牌 （Tokens） 数量（y轴）与交互数量（x轴）的变化：缓冲内存 `ConversationBufferMemory` 与总结内存  `ConversationSummaryMemory` 。

对于较长的对话来说，是的。

[在这里](https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/03a-token-counter.ipynb)，我们有一个较长的对话。

如上所示，总结内存最初使用的令牌 （Tokens） 数量要多得多。

然而，随着对话的进行，总结方法的增长速度较慢。

相比之下，缓冲区内存的令牌 （Tokens） 数量与聊天中的令牌 （Tokens） 数量呈线性增长。

我们总结`ConversationSummaryMemory`的优点和缺点如下：


| 优点 | 缺点 |
| --- | --- |
| 缩短*长*对话的令牌 （Tokens） 数量 | 对于较小的对话，可能会导致更多的令牌 （Tokens） 使用 |
| 能够实现更长的对话 | 对话历史的记忆完全依赖于中间摘要LLM的概括能力 |
| 实现相对简单，直观易懂 | 摘要LLM也需要使用令牌 （Tokens） ，这会增加成本（但不限制对话长度） |

对话摘要是处理长对话的良好方法。然而，它仍然受到令牌 （Tokens） 限制的基本限制。

经过一段时间后，我们仍然会超出上下文窗口的限制。

### 对话缓冲窗口记忆 ConversationBufferWindowMemory 

`ConversationBufferWindowMemory`的功能与我们之前的 `缓冲记忆 ConversationBufferMemory` 相同，但是在记忆中添加了一个 `窗口 Window` 。这意味着在 “忘记” 之前，我们只保留给定数量的过去交互。

我们使用它的方式如下所示：

```python
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

conversation = ConversationChain(
	llm = llm,
	memory = ConversationBufferWindowMemory(k = 1)
)

```
在这个例子中，我们设置`k=1`，这意味着窗口将记住人类和AI之间的单个最新交互。

也就是最新的人类回应和最新的AI回应。我们可以在下面看到其效果：


In[61]:
```python
count_tokens(
    conversation_bufw, 
    "Good morning AI!"
)
```
Out[61]:
```python
Spent a total of 85 tokens
" Good morning! It's a beautiful day today, isn't it? How can I help you?"
```
In[62]:
```python
count_tokens(
    conversation_bufw, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)
```
Out[62]:
```python
Spent a total of 178 tokens
' Interesting! Large Language Models are a type of artificial intelligence that can process natural language and generate text. They can be used to generate text from a given context, or to answer questions about a given context. Integrating them with external knowledge can help them to better understand the context and generate more accurate results. Do you have any specific questions about this integration?'
```
In[63]:
```python
count_tokens(
    conversation_bufw, 
    "I just want to analyze the different possibilities. What can you think of?"
)
```
Out[63]:
```python
Spent a total of 233 tokens
' There are many possibilities for integrating Large Language Models with external knowledge. For example, you could use external knowledge to provide additional context to the model, or to provide additional training data. You could also use external knowledge to help the model better understand the context of a given text, or to help it generate more accurate results.'
```
In[64]:
```python
count_tokens(
    conversation_bufw, 
    "Which data source types could be used to give context to the model?"
)
```
Out[64]:
```python
Spent a total of 245 tokens
' Data sources that could be used to give context to the model include text corpora, structured databases, and ontologies. Text corpora provide a large amount of text data that can be used to train the model and provide additional context. Structured databases provide structured data that can be used to provide additional context to the model. Ontologies provide a structured representation of knowledge that can be used to provide additional context to the model.'
```
In[65]:
```python
count_tokens(
    conversation_bufw, 
    "What is my aim again?"
)
```
Out[65]:
```python
Spent a total of 186 tokens
' Your aim is to use data sources to give context to the model.'
```
在对话结束时，当我们询问：“我的目标是什么？”时，这个答案在三次交互之前的人类回应中包含。

由于我们只保留了最近的交互（`k=1`），模型已经遗忘了，并且无法给出正确的答案。

我们可以这样查看模型的 “记忆” 效果：

In[66]:
```python
bufw_history = conversation_bufw.memory.load_memory_variables(
    inputs = []
)['history']:
```
In[67]:
```python
print(bufw_history)
```
Out[67]:
```python
Human: What is my aim again?
AI:  Your aim is to use data sources to give context to the model.

```

尽管这种方法不适合记忆较远的交互，但它在限制令牌 （Tokens） 使用数量方面表现出色——这是一个我们可以根据需要增加/减少的数字。

对于我们在之前比较中使用的[较长对话](https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/03a-token-counter.ipynb)，我们可以设置`k=6`，在27次总交互后每个交互达到约1.5K个令牌 （Tokens） ：

![包括缓冲窗口记忆的令牌 （Tokens） 数量比较](https://d33wubrfki0l68.cloudfront.net/f076c3f2968833363a7083882c8f648f552cdeb7/23eba/images/langchain-conversational-memory-4.png)包括`ConversationBufferWindowMemory`在`k=6`和`k=12`时的令牌 （Tokens） 数量。

如果我们只需要最近交互的记忆，这是一个很好的选择。然而，对于远程和最近交互的混合，还有其他选择。

### 对话摘要缓冲窗口记忆 ConversationSummaryBufferMemory

`ConversationSummaryBufferMemory`是`ConversationSummaryMemory`和`ConversationBufferWindowMemory`的混合体。

它概述了对话中最早的交互，同时保留其对话中最近的`max_token_limit`个令牌 （Tokens） 。

它的初始化方式如下所示：

```python
conversation_sum_bufw = ConversationChain(
    llm = llm, memory = ConversationSummaryBufferMemory(
        llm = llm,
        max_token_limit = 650
)

```
将其应用于我们之前的对话时，我们可以将`max_token_limit`设为一个较小的数值，但LLM仍然能记住我们之前的 `目标` 。

这是因为该信息被记忆的 `摘要 Summary` 组件捕获，尽管被 `缓冲窗口 BufferWindow` 组件忽略。

当然，该组件的优缺点是基于其所基于的早期组件的综合结果。


| 优点 | 缺点 |
| --- | --- |
|  Summary 摘要器意味着我们可以记住较远的交互 | 摘要器增加了较短对话的令牌 （Tokens） 数量 |
| Buffer 缓冲器防止我们错过最近交互的信息 | 存储原始交互——即使只是最近交互——增加了令牌 （Tokens） 数量 |

尽管需要更多调整以确定摘要和缓冲窗口内要保留的内容，但`ConversationSummaryBufferMemory`确实为我们提供了很大的灵活性，并且是我们目前唯一一个既能记住远程交互又能以其原始形式（最丰富的信息形式）存储最近交互的记忆类型。

![不同对话记忆类型的完整比较及对令牌 （Tokens） 数量的影响](https://d33wubrfki0l68.cloudfront.net/114eadb6f869d328ebc828d018aec8ce1a498437/89d76/images/langchain-conversational-memory-5.png)包括`ConversationSummaryBufferMemory`类型的令牌 （Tokens） 数量比较，其`max_token_limit`值分别为`650`和`1300`。

我们还可以看到，尽管包括了过去交互的摘要和最近交互的原始形式，`ConversationSummaryBufferMemory`的令牌 （Tokens） 数量增加与其他方法相当。

### 其他记忆类型 Other Memory Types

我们在这里介绍的记忆类型非常适合入门，并在在尽可能记住尽可能多的内容和最小化令牌 （Tokens） 之间取得了良好的平衡。

然而，我们还有其他选择——尤其是`ConversationKnowledgeGraphMemory`和`ConversationEntityMemory`。

我们将在接下来的章节中对这些不同形式的记忆进行详细介绍。


---
这就是关于使用LangChain的LLM进行对话记忆的介绍。

正如我们所见，有很多选项可以帮助*无状态 state*的LLM以类似于*有状态 state* 环境的方式进行交互，能够考虑并参考过去的交互。

如前所述，我们还可以涵盖其他形式的记忆。我们还可以实现自己的记忆模块，在同一链中使用多种类型的记忆，将它们与代理 (Agents) 

结合使用等等。所有这些内容将在未来的章节中介绍。


---
[下一章：使用知识库修复幻觉](https://cookbook.langchain.com.cn/docs/langchain-retrieval-augmentation/)
---
