
# *Bag-of-Words* e o modelo Multinomial

## Capítulo 5 - *Bag-of-words*

Na última aula, vimos o que é um córpus e os princípios que o pesquisador deve seguir na construção ou utilização de um córpus para tarefas de aprendizado supervisionado. Depois de construir um córpus de documentos, o pesquisador deve decidir como irá representar o texto de forma numérica. Uma das formas mais comuns, e utilizadas nos modelos mais clássicos de aprendizado estatístico, é a *bag-of-words*, ou saco de palavras. Essa representação será a base dos modelos iniciais do curso que veremos a partir da sexta aula, como o modelo logístico/multinomial. A ideia principal por trás desse modelo é a de representar cada documento pelo número de vezes que cada palavra aparece nele. Considere as seguintes frases (ou documentos):

1. O cachorro ama o osso;
2. O Osso ama o cachorro;

Ignorando o artigo "O", teríamos a seguinte matriz *Document-feature*, ou o seguinte BOW (*Bag-of-words*):


<div align="center">

| Documento | Cachorro | ama | Osso |
|-----------|------|-------|----------|
| Doc1      | 1    | 1     | 1        |
| Doc2      | 1    | 1     | 1        |

</div>

Essa matriz é chamada assim por que cada linha (ou observação) contém um documento/sentença/frase e cada coluna contém a palavra em consideração. Por exemplo, a coluna "Cachorro" mostra quantas vezes o termo Cachorro aparece em cada documento. Neste caso, os dois documentos possuem as mesmas palavras, e seriam tratados como "iguais". Isso se deve ao fato de que, nesse tipo de representação, a ordem das palavras e o contexto não são considerados; por isso, embora as duas frases expressem relações diferentes, a Bag-of-Words as representa de forma idêntica, pois contêm as mesmas palavras com as mesmas contagens. Mais adiante no curso, serão apresentadas outras representações que capturam contexto, como n-gramas, embeddings e modelos contextuais. No entanto, apesar da sua simplicidade e redução do contexto dos documentos, a BOW pode ser extraordinariemente eficiente a depender do seu objetivo e tarefa. Grimmer et al. estabelecem a seguinte "receita" para se trabalhar com a representação BOW:

1. Escolher a unidade de análise;
2. Tokenizar o texto;
3. Reduzir complexidade;
4. Criar a matriz *document-feature*, ou documentos-termos.


### Escolher a unidade de análise

Depois de garantir a digitalização dos documentos, ou coletá-los, o pesquisador deve decidir como vai dividi-los em unidades de análise menores. No nosso exemplo, não precisamos dividir os documentos em unidades menores pois eles já estão em formato reduzido, por serem *tweets* de até 280 caractéres. No entanto, em muitos casos os pesquisadores estarão interessados em textos mais longos, como declarações, discursos, manifestos, notícias, artigos. Em outros casos, o objetivo pode ser o de juntar textos menores em documentos maiores (e.g. juntar os tweets de um dia de determinado usuário). Independente da situação, o pesquisador deve estar consciente da decisão e escolher aquela que seja melhor para atender o seu objetivo de pesquisa. A unidade de análise pode afetar a pergunta de pesquisa, a eficiência dos modelos e a própria coleta de dados. A unidade de análise também é conhecida por *documento*, podendo ser o parágrafo de uma notícia, uma página inteira, ou apenas um *tweet*.



### "Tokenizar"

"Tokenizar", ou transformar em *tokens*, vem depois da escolha da unidade de análise. Cada documento será quebrado em suas partes individuais, as palavras. Esse é o primeiro passo para a criação do saco de palavras (BOW). O exemplo mais comum é o do tokenizar em n-grams de 1. A frase "Diga não à vacinação obrigatória!" pode ser quebrada da seguinte forma:


<div align="center">

["Diga","não","à","vacinação","obrigatória", "!"]

</div>


Seguindo a mesma frase e a mesma etapa de tokenização em palavras, podemos formar bi-grams, que são pares de palavras consecutivas no texto. Bi-grams ajudam a capturar um pouco de contexto local que o unigram (n=1) não representa, como expressões fixas, negações e relações imediatas entre termos.

<div align="center">

["Diga_não", "não_à", "à_vacinação", "vacinação_obrigatória", "obrigatória_!"]

</div>

Se o objetivo é capturar resistência à vacinação, ou hesitação vacinal, talvez quebrar o texto em bigramas seja mais informativo do que quebrá-lo apenas em suas palavras individuais. Com isso, se dá mais contexto ao modelo de aprendizado de máquina, ao custo de adicionar mais combinações raras que podem não aparecer tanto em seu banco de dados. 💬 "Usar ordens maiores de n-gramas pode aumentar substancialmente o número de tipos únicos, mas pode ajudar nossa análise textual ao reter mais informações" ({cite}`grimmer2022text`, p. 99, tradução nossa). Novamente, isso não é uma escolha trivial: Assim como todos os passos e princípios discutidos nas últimas aulas, a forma de processar e representar numericamente o texto altera substancialmente os resultados. O pesquisador deve tentar sempre estar consciente dessas escolhas e relatá-las aos leitores quando necessário. 


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Alternativamente, pesquisadores podem achar útil manter uma lista de certos bigramas e trigramas que eles antecipam ser úteis para sua análise. Por exemplo, Rule, Cointet e Bearman (2015) usam algumas expressões de múltiplas palavras, como national security e local government, em sua análise dos discursos SOTU. Vamos indicar isso visualmente colocando um sublinhado entre duas palavras que estão sendo tratadas como um n-grama. Quando a lista de n-gramas a ser extraída é pequena, isso resolve o problema do vocabulário grande, mas exige uma lista de n-gramas."
({cite}`grimmer2022text`, p. 100, tradução nossa)
```

### Reduzir a complexidade

O que Grimmer et al. chamam de "Reduzir a complexidade" é conhecido de forma mais corriqueira no jargão de aprendizado de máquina como "pré-processamento" do texto. A principal função do pré-processamento é o de reduzir os custos computacionais ao diminuir o texto, eliminando palavras muito recorrentes (e, portanto, pouco informativas), reduzir elas à raiz, etc.

#### 1 - Converter o texto em minúsculo

Converter o texto em minúsculo, especialmente em BOW, é garantir que a mesma palavra em minúsculo e em maiúsculo sejam consideradas iguais. Por exemplo, nas frases:

1. "A decisão da Igreja";
2. "A igreja do bairro"

Temos a palavra igreja com duas definições bem diferentes: "Igreja" se referindo à Igreja católica como um todo, no nível da instituição religiosa. E "igreja" se refere ao prédio, um local físico. Na matriz de documentos, teríamos algo mais ou menos assim (ignorando os artigos e preposições):


<div align="center">


| Documento | decisão | Igreja | igreja | bairro |
|-----------|------|-------|----------| --------- |
| Doc1      | 1    | 1     | 0        | 0         |
| Doc2      | 0    | 0     | 1        | 1         |

</div>


Note que surgem duas colunas distintas (“Igreja” e “igreja”) para a mesma forma lexical, inflando o vocabulário. Aplicando *lowercasing* antes da vetorização, consolidamos as formas:

<div align="center">

| Documento | decisão | igreja | bairro |
|-----------|------|-------|---------|
| Doc1      | 1    | 1     | 0         |
| Doc2      | 0    | 0     | 1         |

</div>

Ao invés de quatro colunas, têm-se três. Há casos em que manter a capitalização é útil (p.ex., NER, distinção de nomes próprios como “Porto” vs. “porto”), mas, em pipelines clássicos de BOW, a normalização para minúsculas costuma ser preferida para reduzir ruído e dimensionalidade. A decisão deve considerar o objetivo da tarefa: se a distinção entre nomes próprios e comuns for relevante, pode ser melhor preservar a capitalização ou criar uma feature específica para “capitalizado”.


#### 2 - Remover pontuação

Na maior parte dos casos, a pontuação específica (e.g. vírgulas, pontos finais, acentos, etc.) não vai adicionar informação relevante ao modelo. Voltando ao exemplo anterior:

<div align="center">

["Diga","não","à","vacinação","obrigatória", "!"]

</div>

Podemos ter:

<div align="center">

["Diga","nao","a","vacinaçao","obrigatoria"]

</div>

Esse processo reduz a quantidade de colunas e termos, facilitando o treinamento do modelo de aprendizado de máquina. Novamente, a decisão de manter certas formas, pontuações e termos deve estar alinhada ao objetivo da tarefa.



#### 3 - Remover *stop words*

Podemos estar interessados também em remover certas palavras que, de tão comuns, não adicionam muita informação ao modelo. Exemplos de palavras "*stop words*" pode ser artigos (a, os, as, os), preposições (de,do,desde,em) ou outras palavras que o pesquisador, ao se defrontar com o texto, considere irrelevantes para a diferenciação dos documentos. Com esse passo adicional, a frase acima se torna:

<div align="center">

["Diga","nao","vacinaçao","obrigatoria"]

</div>

O pesquisador pode utilizar listas comuns de *stop words* em várias línguas, ou criar sua própria lista.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Remover stopwords frequentemente elimina uma fração extremamente alta dos tokens do córpus enquanto remove muito poucos tipos, o que — se essas palavras carregam pouco ou nenhum significado — pode resultar em enormes economias computacionais. Isso reflete o fato empírico de que, na maioria das línguas, poucas palavras são extremamente comuns. A ideia é que, embora esses tokens respondam por uma grande fração das palavras, eles representam apenas uma pequena fração do significado. Listas de stopwords em muitas línguas estão disponíveis em pacotes de software que fornecem ferramentas de análise de texto."
({cite}`grimmer2022text`, p. 102, tradução nossa)
```


#### 4 - Reduzir à raiz

Um outro passo é o de reduzir as palavras às suas raízes, diminuindo os termos únicos nos documentos. “Reduzir à raiz” é um passo de pré-processamento que transforma palavras em uma forma básica para diminuir variações superficiais e, assim, reduzir a esparsidade do vocabulário na matriz documento-termo. Na prática, usa-se:

- **Stemming**: corta sufixos com regras heurísticas para obter um radical aproximado. É rápido, mas pode gerar formas não dicionarizadas (ex.: “estudando”, “estudaram”, “estudarão” → “estud”).

- **Lematização**: mapeia cada palavra para o seu lema (forma canônica), considerando informação morfológica/gramatical. É mais precisa, porém mais custosa (ex.: “estudando”, “estudaram”, “estudarão” → “estudar”). 

Voltando ao exemplo da vacinação obrigatória, teríamos, com o *stemming*:

<div align="center">

["dig","nao","vacina","obrig"]

</div>

Com a lematização, ficaria:

<div align="center">

["dizer","não","vacinação","obrigatório"]


</div>


#### 5 - Eliminar Palavras muito frequentes ou muito raras

Como fica claro, o último passo é o de remover palavras que são muito frequentes ou muito raras. Essas palavras, devido às suas frequências anormais, acrescentam pouca informação ao modelo, gerando barulho desnecessário. Portanto, o pesquisador pode optar por filtrar palavras que seriam "*outliers*" nos documentos.


### Construir a matriz *document-feature*

Depois de todos esses passos, o final é o de transformar o texto em dados numéricos. O resultado final é uma matriz de documentos-termos, chamada de *W*, com N textos e um vocabulário de tamanho (Matriz N x J). No geral, essa matriz é muito esparsa, isto é, contém muitos zeros em muitas colunas para muitos documentos. Por isso é necessário tokenizar, eliminar certos termos, etc. 


### Quando não pré-processar?

Nem sempre vale aplicar todo o pré-processamento “no automático”. Em tarefas que dependem de nuances de forma ou contexto — como reconhecimento de entidades (nomes próprios, marcas, locais), análise estilística, detecção de ironia/sarcasmo, distinções semânticas sensíveis à capitalização (“Porto” vs. “porto”), ou quando a pontuação e emojis carregam sinal (sentimento, ênfase) — remover pontuação, fazer lowercasing agressivo, tirar stopwords ou reduzir à raiz pode apagar informação útil. Além disso, em domínios especializados (jurídico, biomédico), a forma exata e a morfologia costumam importar; já em textos curtos (como títulos e tweets), cada token tende a ter mais peso, e filtros pesados podem “esvaziar” o sinal. Por isso, é recomendável: alinhar cada transformação ao objetivo da tarefa; testar incrementalmente o impacto de cada etapa; manter versões reproduzíveis do pipeline; ajustar listas de stopwords ao domínio; preferir lematização a stemming quando a precisão lexical for crucial; registrar decisões e métricas; e, quando fizer sentido, combinar representações (p.ex., manter capitalização como feature adicional, preservar certos sinais de pontuação/emoji ou incluir n-grams) em vez de eliminar informação de forma irreversível.






