LangChain：介绍与入门===========================================
**L**arge **L**anguage **M**odels（LLMs）于2020年OpenAI的GPT-3发布时登上世界舞台[1]。从那时起，它们在人气方面一直保持稳定的增长。
直到2022年末，LLM和生成AI等广泛领域的兴趣才激增。这一原因很可能是LLM方面的重大进展不断向上推进。
我们看到了有关Google的“有感知能力”的LaMDA聊天机器人的重大新闻。发布了首个高性能且开源的LLM——BLOOM。OpenAI发布了他们的下一代文本嵌入模型和下一代“GPT-3.5”模型。
在LLM领域取得如此巨大的飞跃之后，OpenAI发布了*ChatGPT*，将LLM推向聚光灯下。
[LangChain](https://github.com/hwchase17/langchain)也在同一时期出现。它的创造者Harrison Chase于2022年10月底进行了首次提交。在被卷入LLM浪潮之前，只有短短几个月的开发时间。
尽管这个库还处于早期阶段，但它已经充满了围绕LLM核心构建惊人工具所需的令人难以置信的功能。在本文中，我们将介绍这个库，并从LangChain提供的最简单的组件开始——LLMs。


---
LangChain---------
在其核心，LangChain是围绕LLMs构建的框架。我们可以将其用于聊天机器人，[**G**enerative **Q**uestion-**A**nswering (GQA)](https://www.pinecone.io/learn/openai-gen-qa/)，摘要等等。
该库的核心思想是我们可以将不同的组件*“链”*在一起，以创建更高级的LLMs用例。链可能由来自几个模块的多个组件组成：
* **Prompt templates**：Prompt templates是不同类型提示的模板。例如“chatbot”样式模板、ELI5问答等* **LLMs**：像GPT-3、BLOOM等大型语言模型* **Agents**：Agents使用LLMs决定应采取的操作。可以使用诸如网络搜索或计算器之类的工具，并将所有工具包装成一个逻辑循环的操作。* **Memory**：短期记忆、长期记忆。
我们将在LangChain手册的即将推出的章节中详细介绍每个组件。您可以通过我们的通讯订阅保持每个版本的更新：
如果你是人类，请勿填写此项：订阅以获取LangChain的最新消息！
提交订阅成功。提交失败。现在，我们将从**提示模板**和**LLMs**的基础知识开始。我们还将探索库中提供的两个LLMs选项，使用来自*Hugging Face Hub*或*OpenAI*的模型。
我们的第一个Prompt模板--------------------------
将输入到LLMs的提示通常具有不同的结构，以便我们可以获得不同的结果。对于问答，我们可以将用户的问题重新格式化为不同的问答样式，例如传统问答、答案的项目列表，甚至是与给定问题相关的问题摘要。
### 在LangChain中创建提示
让我们组合一个简单的问答提示模板。我们首先需要安装`langchain`库。

```
!pip install langchain

```


---
*通过[Colab](https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/langchain/handbook/00-langchain-intro.ipynb)与代码一起进行操作！*


---
从这里，我们导入`PromptTemplate`类并初始化一个模板，如下所示：

```
from langchain import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

```
当使用这些提示模板与给定的`question`时，我们将获得：

```
Question: Which NFL team won the Super Bowl in the 2010 season?

Answer: 

```
目前为止，这就是我们所需要的全部内容。我们将在Hugging Face Hub和OpenAI LLM生成中使用相同的提示模板。
Hugging Face Hub LLM--------------------
LangChain中的Hugging Face Hub端点连接到Hugging Face Hub，并通过其免费推断端点运行模型。我们需要一个[Hugging Face帐户和API密钥](https://huggingface.co/settings/tokens)来使用这些端点。
获得API密钥后，我们将其添加到`HUGGINGFACEHUB_API_TOKEN`环境变量中。我们可以使用Python来做到这一点：

```
import os

os.environ['HUGGINGFACEHUB\_API\_TOKEN'] = 'HF\_API\_KEY'

```
然后，我们必须通过Pip安装`huggingface_hub`库。

```
!pip install huggingface_hub

```
现在，我们可以使用Hub模型生成文本。我们将使用[`google/flan-t5-x1`](https://huggingface.co/google/flan-t5-xl)。


---
*默认的Hugging Face Hub推断API不使用专用硬件，因此速度较慢。它们也不适用于运行较大模型，例如`bigscience/bloom-560m`或`google/flan-t5-xxl`（请注意`xxl`与`xl`的区别）*


---
In[3]:```
from langchain import HuggingFaceHub, LLMChain

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':1e-10}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about NFL 2010
print(llm_chain.run(question))
```
Out[3]:```
green bay packers

```
对于这个问题，我们得到了`"green bay packers"`的正确答案。
### 提出多个问题
如果我们想要提出多个问题，我们可以尝试两种方法：
1. 使用`generate`方法遍历所有问题，逐个回答。2. 将所有问题放入单个提示中，这仅适用于更高级的LLMs。
从选项（1）开始，让我们看看如何使用`generate`方法：
In[4]:```
qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
res = llm_chain.generate(qs)
res
```
Out[4]:```
LLMResult(generations=[[Generation(text='green bay packers', generation_info=None)], [Generation(text='184', generation_info=None)], [Generation(text='john glenn', generation_info=None)], [Generation(text='one', generation_info=None)]], llm_output=None)
```
除了第一个问题外，我们得到的结果都很糟糕。这只是所使用的LLM的局限性。
如果模型无法准确回答单个问题，将所有查询组合到一个提示中很可能不起作用。不过，为了进行实验，让我们试试。
In[6]:```
multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=flan_t5
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))
```
Out[6]:```
If I am 6 ft 4 inches, how tall am I in centimeters

```
正如预期的那样，结果并不有用。后面我们会看到更强大的LLMs可以做到这一点。
OpenAI LLMs-----------
LangChain中的OpenAI端点直接连接到OpenAI或通过Azure。我们需要一个[OpenAI帐户和API密钥](https://beta.openai.com/account/api-keys)来使用这些端点。
获得API密钥后，我们将其添加到`OPENAI_API_TOKEN`环境变量中。我们可以使用Python来做到这一点：

```
import os

os.environ['OPENAI\_API\_TOKEN'] = 'OPENAI\_API\_KEY'

```
然后，我们必须通过Pip安装`openai`库。

```
!pip install openai

```
现在，我们可以使用OpenAI的GPT-3生成（或*完成*）模型生成文本。我们将使用[`text-davinci-003`](https://huggingface.co/google/flan-t5-xl)。

```
from langchain.llms import OpenAI

davinci = OpenAI(model_name='text-davinci-003')

```


---
*或者，如果您使用的是Azure上的OpenAI，则可以执行以下操作：*

```
from langchain.llms import AzureOpenAI

llm = AzureOpenAI(
    deployment_name="your-azure-deployment", 
    model_name="text-davinci-003"
)

```


---
我们将与Hugging Face示例一样使用相同的简单问答提示模板。唯一的变化是我们现在传递我们的OpenAI LLM `davinci`：
In[15]:```
llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(question))
```
Out[15]:```
 The Green Bay Packers won the Super Bowl in the 2010 season.

```
正如预期的那样，我们得到了正确的答案。我们可以使用`generate`来处理多个问题：
In[16]:```
qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
llm_chain.generate(qs)
```
Out[16]:```
LLMResult(generations=[[Generation(text=' The Green Bay Packers won the Super Bowl in the 2010 season.', generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text=' 193.04 centimeters', generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text=' Charles Duke was the 12th person on the moon. He was part of the Apollo 16 mission in 1972.', generation_info={'finish_reason': 'stop', 'logprobs': None})], [Generation(text=' A blade of grass does not have any eyes.', generation_info={'finish_reason': 'stop', 'logprobs': None})]], llm_output={'token_usage': {'total_tokens': 124, 'prompt_tokens': 75, 'completion_tokens': 49}})
```
我们大部分的结果都是正确的或有一定的真实性。与`google/flan-t5-xl`模型相比，该模型的性能显然更好。与之前一样，让我们尝试将所有问题一次性输入模型。
In[17]:```
llm_chain = LLMChain(
    prompt=long_prompt,
    llm=davinci
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))
```
Out[17]:```
The New Orleans Saints won the Super Bowl in the 2010 season.

```

```
6 ft 4 inches is 193 centimeters.

```

```
The 12th person on the moon was Harrison Schmitt.

```

```
A blade of grass does not have eyes.

```
当我们反复运行查询时，模型偶尔会出错，但有时也能正确回答所有问题。


---
这就是我们对LangChain的介绍了——这是一个使我们能够构建更高级应用程序的库，这些应用程序围绕OpenAI的GPT-3模型或通过Hugging Face提供的开源替代方案。
正如前面提到的，LangChain可以做的远不止我们在这里演示的。我们将在即将发布的文章中介绍这些其他功能。


---
参考资料----------
[1] [GPT-3 存档库](https://github.com/openai/gpt-3) (2020), OpenAI GitHub


---
[下一章：使用 Langchain 进行提示工程和 LLMs](/learn/langchain-prompt-templates/)
---
### 评论
const pageURL=window.location.protocol+'//'+window.location.host+window.location.pathnameDiscourseEmbed={discourseUrl:"https://community.pinecone.io/",discourseEmbedUrl:pageURL,};(function(){var d=document.createElement("script");d.type="text/javascript";d.async=true;d.src=DiscourseEmbed.discourseUrl+"javascripts/embed.js";(document.getElementsByTagName("head")[0]||document.getElementsByTagName("body")[0]).appendChild(d);})();