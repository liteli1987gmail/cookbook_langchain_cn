修复知识库中的幻觉 
=========================================

Fixing Hallucination with Knowledge Bases
---


**大型语言模型（LLMs）**存在数据实时性的问题。即使是像 GPT-4 这样最强大的模型也对最近的事件一无所知。

在 LLMs 看来，世界是静止的。它们只知道通过它们的训练数据所呈现的世界。

这给依赖最新信息或特定数据集的任何用例带来了问题。例如，您可能有一些内部公司文档，您希望通过 LLMs 与之交互。

第一个挑战是将这些文档添加到 LLMs 中，我们可以尝试对这些文档进行训练，但这是耗时且昂贵的。

当添加新文档时会发生什么？对于每个新添加的文档进行训练是非常低效的，简直是不可能的。

那么，我们该如何解决这个问题呢？我们可以使用“检索增强 retrieval augmentation ”的技术。这种技术允许我们从外部知识库中检索相关信息并将该信息提供给我们的 LLMs。

外部知识库是我们窥视 LLMs 训练数据之外世界的“窗口 window ”。在本章中，我们将学习如何使用 LangChain 为 LLMs 实现检索增强。


创建知识库
---------------------------

对于 LLMs，我们有两种主要类型的知识。 “参数化知识 parametric knowledge ”是指 LLMs 在训练过程中学到的一切，它作为 LLMs 对世界的冻结快照。

第二种类型的知识是“源知识 source knowledge”。这种知识涵盖通过输入提示输入 LLMs 的任何信息。当我们谈论“检索增强”时，我们是指给 LLMs 提供有价值的源知识。

*（您可以使用 [此处的 Jupyter 笔记本](https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/05-langchain-retrieval-augmentation.ipynb) 跟随以下章节！）*

### 获取我们的知识库数据 Getting Data for our Knowledge Base

为了帮助我们的 LLMs，我们需要为其提供相关的源知识。为此，我们需要创建我们的知识库。

我们从一个数据集开始。使用的数据集自然取决于用例。

它可以是 LLMs 需要帮助编写代码的代码文档，也可以是内部聊天机器人的公司文档，或者其他任何东西。

在我们的示例中，我们将使用维基百科的一个子集。要获取这些数据，我们将使用 Hugging Face 数据集，如下所示：


In [2]:
```python
from datasets import load_dataset

data = load_dataset("wikipedia", "20220301.simple", split ='train [: 10000]')
data
```
Out[2]:
```python
Downloading readme:   0%|          | 0.00/16.3k [00: 00 <?, ?B/s]

Dataset({
    features: ['id', 'url', 'title', 'text'],
    num_rows: 10000
})
```

In[3]:
```python
data [6]
```
Out[3]:
```python
{'id': '13',

 'url': 'https://simple.wikipedia.org/wiki/Alan%20Turing',

 'title': 'Alan Turing',

 'text': 'Alan Mathison Turing OBE FRS (London, 23 June 1912 – Wilmslow, Cheshire, 7 June 1954) was an English mathematician and computer scientist. He was born in Maida Vale, London.\n\nEarly life and family \nAlan Turing was born in Maida Vale, London on 23 June 1912. His father was part of a family of merchants from Scotland. His mother, Ethel Sara, was the daughter of an engineer.\n\nEducation \nTuring went to St. Michael\'s, a school at 20 Charles Road, St Leonards-on-sea, when he was five years old.\n "This is only a foretaste of what is to come, and only the shadow of what is going to be.” – Alan Turing.\n\nThe Stoney family were once prominent landlords, here in North Tipperary. His mother Ethel Sara Stoney (1881–1976) was daughter of Edward Waller Stoney (Borrisokane, North Tipperary) and Sarah Crawford (Cartron Abbey, Co. Longford); Protestant Anglo-Irish gentry.\n\nEducated in Dublin at Alexandra School and College; on October 1st 1907 she married Julius Mathison Turing, latter son of Reverend John Robert Turing and Fanny Boyd, in Dublin. Born on June 23rd 1912, Alan Turing would go on to be regarded as one of the greatest figures of the twentieth century.\n\nA brilliant mathematician and cryptographer Alan was to become the founder of modern-day computer science and artificial intelligence; designing a machine at Bletchley Park to break secret Enigma encrypted messages used by the Nazi German war machine to protect sensitive commercial, diplomatic and military communications during World War 2. Thus, Turing made the single biggest contribution to the Allied victory in the war against Nazi Germany, possibly saving the lives of an estimated 2 million people, through his effort in shortening World War II.\n\nIn 2013, almost 60 years later, Turing received a posthumous Royal Pardon from Queen Elizabeth II. Today, the “Turing law” grants an automatic pardon to men who died before the law came into force, making it possible for living convicted gay men to seek pardons for offences now no longer on the statute book.\n\nAlas, Turing accidentally or otherwise lost his life in 1954, having been subjected by a British court to chemical castration, thus avoiding a custodial sentence. He is known to have ended his life at the age of 41 years, by eating an apple laced with cyanide.\n\nCareer \nTuring was one of the people who worked on the first computers. He created the theoretical  Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.\n\nTuring was interested in artificial intelligence. He proposed the Turing test, to say when a machine could be called " intelligent ". A computer could be said to " think " if a human talking with it could not tell it was a machine.\n\nDuring World War II, Turing worked with others to break German ciphers (secret messages). He  worked for the Government Code and Cypher School (GC&CS) at Bletchley Park, Britain\'s codebreaking centre that produced Ultra intelligence.\nUsing cryptanalysis, he helped to break the codes of the Enigma machine. After that, he worked on other German codes.\n\nFrom 1945 to 1947, Turing worked on the design of the ACE (Automatic Computing Engine) at the National Physical Laboratory. He presented a paper on 19 February 1946. That paper was " the first detailed design of a stored-program computer ". Although it was possible to build ACE, there were delays in starting the project. In late 1947 he returned to Cambridge for a sabbatical year. While he was at Cambridge, the Pilot ACE was built without him. It ran its first program on 10\xa0May 1950.\n\nPrivate life \nTuring was a homosexual man. In 1952, he admitted having had sex with a man in England. At that time, homosexual acts were illegal. Turing was convicted. He had to choose between going to jail and taking hormones to lower his sex drive. He decided to take the hormones. After his punishment, he became impotent. He also grew breasts.\n\nIn May 2012, a private member\'s bill was put before the House of Lords to grant Turing a statutory pardon. In July 2013, the government supported it. A royal pardon was granted on 24 December 2013.\n\nDeath \nIn 1954, Turing died from cyanide poisoning. The cyanide came from either an apple which was poisoned with cyanide, or from water that had cyanide in it. The reason for the confusion is that the police never tested the apple for cyanide. It is also suspected that he committed suicide.\n\nThe treatment forced on him is now believed to be very wrong. It is against medical ethics and international laws of human rights. In August 2009, a petition asking the British Government to apologise to Turing for punishing him for being a homosexual was started. The petition received thousands of signatures. Prime Minister Gordon Brown acknowledged the petition. He called Turing\'s treatment " appalling ".\n\nReferences\n\nOther websites \nJack Copeland 2012. Alan Turing: The codebreaker who saved \'millions of lives\'. BBC News / Technology \n\nEnglish computer scientists\nEnglish LGBT people\nEnglish mathematicians\nGay men\nLGBT scientists\nScientists from London\nSuicides by poison\nSuicides in the United Kingdom\n1912 births\n1954 deaths\nOfficers of the Order of the British Empire'}
```

大多数数据集将包含包含*大量*文本的记录。

因此，我们的第一个任务通常是构建一个预处理流水线，将这些长文本切割成更*简洁*的块 Chunks。

### 创建块 Creating Chunks

将我们的文本分割成较小的块对于多个原因非常重要。主要有以下几点：
* 提高 “ 嵌入 （Embeddings） 准确性 embedding accuracy ” - 这将提高后续结果的相关性。
* 减少输入LLMs的文本量。限制输入可以提高LLMs遵循指示的能力，减少生成成本，并帮助我们获得更快的响应。
* 为用户提供更精确的信息源，因为我们可以将信息源缩小到更小的文本块。
* 对于*非常长*的文本块，我们将超过嵌入 （Embeddings） 或完成模型的最大上下文窗口。将这些较长的文档拆分使得可以将这些文档添加到我们的知识库中。

为了创建这些块，我们首先需要一种衡量文本长度的方法。LLMs不是按单词或字符来衡量文本的-它们是按“令牌 （Tokens） ”来衡量的。

令牌 （Tokens） 通常是一个词或子词的大小，根据LLMs而异。这些令牌 （Tokens） 本身是使用“标记器 （Tokenizer） ”构建的。

我们将使用`gpt-3.5-turbo`作为我们的模型，并且我们可以像下面这样初始化该模型的标记器 （Tokenizer） ：

```python
import tiktoken  # ! pip install tiktoken

tokenizer = tiktoken.get_encoding('p50k_base')

```

使用标记器 （Tokenizer） ，我们可以从纯文本创建标记并计算标记数。我们将将此封装为一个名为`tiktoken_len`的函数：


In[28]:
```python
# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special =()
    )
    return len(tokens)

tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
             "we can find the length of this chunk of text in tokens")
```
Out[28]:
```python
28
```
有了我们的标记计数函数，我们可以初始化一个LangChain `RecursiveCharacterTextSplitter`对象。该对象将允许我们将文本分割成不超过我们通过`chunk_size`参数指定的长度的块。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 20,
    length_function = tiktoken_len,
    separators = ["\n\n", "\n", " ", ""]
)

```
现在我们这样分割文本：


In[6]:
```python
chunks = text_splitter.split_text(data [6]['text'])[: 3]
chunks
```
Out[6]:
```python
['Alan Mathison Turing OBE FRS (London, 23 June 1912 – Wilmslow, Cheshire, 7 June 1954) was an English mathematician and computer scientist. He was born in Maida Vale, London.\n\nEarly life and family \nAlan Turing was born in Maida Vale, London on 23 June 1912. His father was part of a family of merchants from Scotland. His mother, Ethel Sara, was the daughter of an engineer.\n\nEducation \nTuring went to St. Michael\'s, a school at 20 Charles Road, St Leonards-on-sea, when he was five years old.\n " This is only a foretaste of what is to come, and only the shadow of what is going to be.” – Alan Turing.\n\nThe Stoney family were once prominent landlords, here in North Tipperary. His mother Ethel Sara Stoney (1881–1976) was daughter of Edward Waller Stoney (Borrisokane, North Tipperary) and Sarah Crawford (Cartron Abbey, Co. Longford); Protestant Anglo-Irish gentry.\n\nEducated in Dublin at Alexandra School and College; on October 1st 1907 she married Julius Mathison Turing, latter son of Reverend John Robert Turing and Fanny Boyd, in Dublin. Born on June 23rd 1912, Alan Turing would go on to be regarded as one of the greatest figures of the twentieth century.\n\nA brilliant mathematician and cryptographer Alan was to become the founder of modern-day computer science and artificial intelligence; designing a machine at Bletchley Park to break secret Enigma encrypted messages used by the Nazi German war machine to protect sensitive commercial, diplomatic and military communications during World War 2. Thus, Turing made the single biggest contribution to the Allied victory in the war against Nazi Germany, possibly saving the lives of an estimated 2 million people, through his effort in shortening World War II.',


 'In 2013, almost 60 years later, Turing received a posthumous Royal Pardon from Queen Elizabeth II. Today, the “Turing law” grants an automatic pardon to men who died before the law came into force, making it possible for living convicted gay men to seek pardons for offences now no longer on the statute book.\n\nAlas, Turing accidentally or otherwise lost his life in 1954, having been subjected by a British court to chemical castration, thus avoiding a custodial sentence. He is known to have ended his life at the age of 41 years, by eating an apple laced with cyanide.\n\nCareer \nTuring was one of the people who worked on the first computers. He created the theoretical  Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.\n\nTuring was interested in artificial intelligence. He proposed the Turing test, to say when a machine could be called "intelligent". A computer could be said to "think" if a human talking with it could not tell it was a machine.\n\nDuring World War II, Turing worked with others to break German ciphers (secret messages). He  worked for the Government Code and Cypher School (GC&CS) at Bletchley Park, Britain\'s codebreaking centre that produced Ultra intelligence.\nUsing cryptanalysis, he helped to break the codes of the Enigma machine. After that, he worked on other German codes.',


 'From 1945 to 1947, Turing worked on the design of the ACE (Automatic Computing Engine) at the National Physical Laboratory. He presented a paper on 19 February 1946. That paper was "the first detailed design of a stored-program computer". Although it was possible to build ACE, there were delays in starting the project. In late 1947 he returned to Cambridge for a sabbatical year. While he was at Cambridge, the Pilot ACE was built without him. It ran its first program on 10\xa0May 1950.\n\nPrivate life \nTuring was a homosexual man. In 1952, he admitted having had sex with a man in England. At that time, homosexual acts were illegal. Turing was convicted. He had to choose between going to jail and taking hormones to lower his sex drive. He decided to take the hormones. After his punishment, he became impotent. He also grew breasts.\n\nIn May 2012, a private member\'s bill was put before the House of Lords to grant Turing a statutory pardon. In July 2013, the government supported it. A royal pardon was granted on 24 December 2013.\n\nDeath \nIn 1954, Turing died from cyanide poisoning. The cyanide came from either an apple which was poisoned with cyanide, or from water that had cyanide in it. The reason for the confusion is that the police never tested the apple for cyanide. It is also suspected that he committed suicide.\n\nThe treatment forced on him is now believed to be very wrong. It is against medical ethics and international laws of human rights. In August 2009, a petition asking the British Government to apologise to Turing for punishing him for being a homosexual was started. The petition received thousands of signatures. Prime Minister Gordon Brown acknowledged the petition. He called Turing\'s treatment "appalling".\n\nReferences\n\nOther websites \nJack Copeland 2012. Alan Turing: The codebreaker who saved \'millions of lives\'. BBC News / Technology']
```
这些块没有超过我们之前设置的400个块大小限制：

In[7]:
```python
tiktoken_len(chunks [0]), tiktoken_len(chunks [1]), tiktoken_len(chunks [2])
```
Out[7]:
```python
(397, 304, 399)
```

使用`text_splitter`，我们可以得到大小合适的文本块。我们将在后面的索引过程中使用此功能。现在，让我们来看看*嵌入 （Embeddings） *。

### 创建嵌入 （Embeddings） 

向我们的LLMs检索相关上下文非常重要的是矢量嵌入 （Embeddings） 。我们将希望将我们想要存储在知识库中的文本块编码为矢量嵌入 （Embeddings） 。

这些嵌入 （Embeddings） 可以作为每个文本块含义的“数值表示”。这是可能的，因为我们使用另一个AI语言模型将人类可读文本转换为AI可读的嵌入 （Embeddings） 。

![编码器将纯文本转换为嵌入 （Embeddings） ](https://d33wubrfki0l68.cloudfront.net/cd74e72b53f29b372f75895b9153e3a56f5fea25/3c7bf/images/langchain-retrieval-augmentation-2.png)

然后，我们将这些嵌入 （Embeddings） 存储在我们的矢量数据库 （Vector Database） 中（稍后详细介绍），并且可以通过计算矢量空间中嵌入 （Embeddings） 之间的距离来找到具有相似含义的文本块。

![矢量空间中具有相似句子的相似区域](https://d33wubrfki0l68.cloudfront.net/a0bc9b1a9ab2e0ff22afb1d33ce1d9875bd0c64a/9af64/images/langchain-retrieval-augmentation-1.png)

我们将使用的嵌入 （Embeddings） 模型是另一个名为`text-embedding-ada-002`的OpenAI模型。

我们可以通过LangChain这样初始化它：

```python
from langchain.embeddings.openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    document_model_name = model_name,
    query_model_name = model_name,
    openai_api_key = OPENAI_API_KEY
)

```
现在我们可以嵌入 （Embeddings） 我们的文本：


In[10]:
```python
texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embed.embed_documents(texts)
len(res), len(res [0])
```
Out[10]:
```python
(2, 1536)
```
从中，我们得到了*两个*嵌入 （Embeddings） ，因为我们传入了两个文本块。

每个嵌入 （Embeddings） 都是一个*1536维*的向量。这个维度只是`text-embedding-ada-002`的输出维度。

有了这些，我们拥有了我们的数据集、文本分割器和嵌入 （Embeddings） 模型。我们拥有了开始构建我们的知识库所需的一切。

### 矢量数据库 （Vector Database）

矢量数据库 （Vector Database） 是一种知识库类型，允许我们将相似嵌入 （Embeddings） 的搜索扩展到数十亿条记录，通过添加、更新或删除记录来管理我们的知识库，甚至可以执行过滤等操作。

我们将使用Pinecone矢量数据库 （Vector Database） 。要使用它，我们需要一个[免费的API密钥](https://app.pinecone.io/)。然后我们可以像这样初始化我们的数据库索引：

```python
import pinecone


Index_name = 'langchain-retrieval-augmentation'

pinecone.init(
        api_key =" YOUR_API_KEY ",  # find api key in console at app.pinecone.io
        environment =" YOUR_ENV "  # find next to api key in console
)

# we create a new index
pinecone.create_index(
        name = index_name,
        metric ='dotproduct',
        dimension = len(res [0]) # 1536 dim of text-embedding-ada-002
)

```
然后我们连接到新索引：


In[12]:
```python

Index = pinecone.GRPCIndex(index_name)


Index.describe_index_stats()
```
Out[12]:
```python
{'dimension': 1536,


 'index_fullness': 0.0,


 'namespaces': {},


 'total_vector_count': 0}
```
我们将看到新的 Pinecone 索引的`total_vector_count`为`0`，因为我们尚未添加任何向量。我们的下一个任务是执行此操作。

索引过程包括遍历我们想要添加到知识库中的数据，创建ID、嵌入 （Embeddings） 和元数据，然后将它们添加到索引中。

我们可以批量处理此过程以加快速度。

```python
from tqdm.auto import tqdm
from uuid import uuid4

batch_limit = 100

texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):
    # first get metadata fields for this record
    metadata = {
        'wiki-id': str(record ['id']),
        'source': record ['url'],
        'title': record ['title']
    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record ['text'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, \*\* metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors = zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

```
现在我们已经对所有内容进行了索引。要检查索引中的记录数，我们再次调用`describe_index_stats`：


In[14]:
```python

Index.describe_index_stats()
```
Out[14]:
```python
{'dimension': 1536,


 'index_fullness': 0.1,


 'namespaces': {'': {'vector_count': 27437}},


 'total_vector_count': 27437}
```
我们的索引包含约27K条记录。如前所述，我们可以将其扩展到数十亿条，但对于我们的示例来说，27K已经足够。

LangChain矢量存储与查询
-----------------------------------
我们独立构建我们的索引，与LangChain无关。因为这是一个直接的过程，并且使用Pinecone客户端直接完成速度更快。

但是，我们要回到LangChain，所以我们应该通过LangChain库重新连接到我们的索引。

```python
from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain

Index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

```
我们可以使用`similarity search`方法直接进行查询，并返回文本块，而无需LLM生成响应。

In[16]:
```python
query = "who was Benito Mussolini?"

vectorstore.similarity_search(
    query,  # our search query
    k = 3  # return 3 most relevant docs
)
```
Out[16]:
```python
[Document(page_content ='Benito Amilcare Andrea Mussolini KSMOM GCTE (29 July 1883 – 28 April 1945) was an Italian politician and journalist. He was also the Prime Minister of Italy from 1922 until 1943. He was the leader of the National Fascist Party.\n\nBiography\n\nEarly life\nBenito Mussolini was named after Benito Juarez, a Mexican opponent of the political power of the Roman Catholic Church, by his anticlerical (a person who opposes the political interference of the Roman Catholic Church in secular affairs) father. Mussolini\'s father was a blacksmith. Before being involved in politics, Mussolini was a newspaper editor (where he learned all his propaganda skills) and elementary school teacher.\n\nAt first, Mussolini was a socialist, but when he wanted Italy to join the First World War, he was thrown out of the socialist party. He \'invented\' a new ideology, Fascism, much out of Nationalist\xa0and Conservative views.\n\nRise to power and becoming dictator\nIn 1922, he took power by having a large group of men, "Black Shirts," march on Rome and threaten to take over the government. King Vittorio Emanuele III gave in, allowed him to form a government, and made him prime minister. In the following five years, he gained power, and in 1927 created the OVRA, his personal secret police force. Using the agency to arrest, scare, or murder people against his regime, Mussolini was dictator\xa0of Italy by the end of 1927. Only the King and his own Fascist party could challenge his power.', lookup_str ='', metadata ={'chunk': 0.0, 'source': 'https://simple.wikipedia.org/wiki/Benito%20Mussolini', 'title': 'Benito Mussolini', 'wiki-id': '6754'}, lookup_index = 0),


 Document(page_content ='Fascism as practiced by Mussolini\nMussolini\'s form of Fascism, "Italian Fascism"- unlike Nazism, the racist ideology that Adolf Hitler followed- was different and less destructive than Hitler\'s. Although a believer in the superiority of the Italian nation and national unity, Mussolini, unlike Hitler, is quoted "Race? It is a feeling, not a reality. Nothing will ever make me believe that biologically pure races can be shown to exist today".\n\nMussolini wanted Italy to become a new Roman Empire. In 1923, he attacked the island of Corfu, and in 1924, he occupied the city state of Fiume. In 1935, he attacked the African country Abyssinia (now called Ethiopia). His forces occupied it in 1936. Italy was thrown out of the League of Nations because of this aggression. In 1939, he occupied the country Albania. In 1936, Mussolini signed an alliance with Adolf Hitler, the dictator of Germany.\n\nFall from power and death\nIn 1940, he sent Italy into the Second World War on the side of the Axis countries. Mussolini attacked Greece, but he failed to conquer it. In 1943, the Allies landed in Southern Italy. The Fascist party and King Vittorio Emanuel III deposed Mussolini and put him in jail, but he was set free by the Germans, who made him ruler of the Italian Social Republic puppet state which was in a small part of Central Italy. When the war was almost over, Mussolini tried to escape to Switzerland with his mistress, Clara Petacci, but they were both captured and shot by partisans. Mussolini\'s dead body was hanged upside-down, together with his mistress and some of Mussolini\'s helpers, on a pole at a gas station in the village of Millan, which is near the border  between Italy and Switzerland.', lookup_str ='', metadata ={'chunk': 1.0, 'source': 'https://simple.wikipedia.org/wiki/Benito%20Mussolini', 'title': 'Benito Mussolini', 'wiki-id': '6754'}, lookup_index = 0),


 Document(page_content ='Fascist Italy \nIn 1922, a new Italian government started. It was ruled by Benito Mussolini, the leader of Fascism in Italy. He became head of government and dictator, calling himself "Il Duce" (which means "leader" in Italian). He became friends with German dictator Adolf Hitler. Germany, Japan, and Italy became the Axis Powers. In 1940, they entered World War II together against France, Great Britain, and later the Soviet Union. During the war, Italy controlled most of the Mediterranean Sea.\n\nOn July 25, 1943, Mussolini was removed by the Great Council of Fascism. On September 8, 1943, Badoglio said that the war as an ally of Germany was ended. Italy started fighting as an ally of France and the UK, but Italian soldiers did not know whom to shoot. In Northern Italy, a movement called Resistenza started to fight against the German invaders. On April 25, 1945, much of Italy became free, while Mussolini tried to make a small Northern Italian fascist state called the Republic of Sal ò. The fascist state failed and Mussolini tried to flee to Switzerland and escape to Francoist Spain, but he was captured by Italian partisans. On 28 April 1945 Mussolini was executed by a partisan.\n\nAfter World War Two \n\nThe state became a republic on June 2, 1946. For the first time, women were able to vote. Italian people ended the Savoia dynasty and adopted a republic government.\n\nIn February 1947, Italy signed a peace treaty with the Allies. They lost all the colonies and some territorial areas (Istria and parts of Dalmatia).\n\nSince then Italy has joined NATO and the European Community (as a founding member). It is one of the seven biggest industrial economies in the world.\n\nTransportation \n\nThe railway network in Italy totals . It is the 17th longest in the world. High speed trains include ETR-class trains which travel at .', lookup_str ='', metadata ={'chunk': 5.0, 'source': 'https://simple.wikipedia.org/wiki/Italy', 'title': 'Italy', 'wiki-id': '363'}, lookup_index = 0)]
```
所有这些都是相关的结果，告诉我们我们的系统的检索组件正在发挥作用。

下一步是添加我们的LLM，利用这些检索到的上下文信息来生成答案。

### 生成式问答

在生成式问答（GQA）中，我们将问题传递给LLM，但指示它基于从知识库返回的信息来回答问题。我们可以在LangChain中轻松实现这一点，使用`RetrievalQA`链。

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key = OPENAI_API_KEY,
    model_name ='gpt-3.5-turbo',
    temperature = 0.0
)

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever()
)

```
让我们尝试使用我们之前的查询:


In[22]:
```python
qa.run(query)
```
Out[22]:
```python
'Benito Mussolini was an Italian politician and journalist who served as the Prime Minister of Italy from 1922 until 1943. He was the leader of the National Fascist Party and invented the ideology of Fascism. Mussolini was a dictator of Italy by the end of 1927, and his form of Fascism, "Italian Fascism," was different and less destructive than Hitler\'s Nazism. Mussolini wanted Italy to become a new Roman Empire and attacked several countries, including Abyssinia (now called Ethiopia) and Greece. He was removed from power in 1943 and was executed by Italian partisans in 1945.'
```
这次我们得到的响应是由我们的`gpt-3.5-turbo` LLM根据从我们的向量数据库检索到的信息生成的。

我们仍然无法完全防止模型产生令人信服但错误的幻觉，这种情况可能发生，并且我们不太可能完全消除这个问题。然而，我们可以采取更多措施来提高对所提供答案的信任。

这样做的一种有效方法是在响应中添加引用，允许用户看到信息的*来源*。

我们可以使用`RetrievalQAWithSourcesChain`的稍微不同版本来实现这一点。


In[23]:
```python
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vectorstore.as_retriever()
)
```

In[24]:
```python
qa_with_sources(query)
```
Out[24]:
```python
{'question': 'who was Benito Mussolini?',


 'answer': 'Benito Mussolini was an Italian politician and journalist who was the Prime Minister of Italy from 1922 until 1943. He was the leader of the National Fascist Party and invented the ideology of Fascism. He became dictator of Italy by the end of 1927 and was friends with German dictator Adolf Hitler. Mussolini attacked Greece and failed to conquer it. He was removed by the Great Council of Fascism in 1943 and was executed by a partisan on April 28, 1945. After the war, several Neo-Fascist movements have had success in Italy, the most important being the Movimento Sociale Italiano. His granddaughter Alessandra Mussolini has outspoken views similar to Fascism. \n',


 'sources': 'https://simple.wikipedia.org/wiki/Benito%20Mussolini, https://simple.wikipedia.org/wiki/Fascism'}
```

现在我们已经回答了提出的问题，同时还包括了LLM使用的信息的*来源*。


---

我们已经学会了如何通过使用矢量数据库 （Vector Database） 作为知识库来为**大型语言模型（LLMs）**提供源知识的支撑。

通过这样做，我们可以鼓励LLM在回答中保持准确性，保持源知识的最新状态，并通过为每个答案提供引用来提高对我们系统的信任。

我们已经看到LLMs和知识库在诸如Bing的AI搜索、Google Bard和[ChatGPT插件](https://youtu.be/hpePPqKxNq8)等大型产品中配对使用。

毫无疑问，LLMs的未来与高性能、可扩展和可靠的知识库密切相关。


---
[下一章：在 LangChain 中使用对话代理 (Agents) 实现超能力 LLMs](https://cookbook.langchain.com.cn/docs/langchain-agents/)
---
