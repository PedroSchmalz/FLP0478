# Modelos de Língua, N-Grams e Classificação com Texto

Na última aula, vimos que o problema de classificação exige métodos específicos quando a variável resposta é categórica, como sentimentos, diagnósticos ou posicionamentos. Discutimos por que a regressão linear não é adequada para esse tipo de tarefa e exploramos alternativas como a regressão logística, que modela probabilidades de forma apropriada. Aprendemos sobre extensões da regressão logística para múltiplos preditores e múltiplas classes, além de conhecer os modelos generativos, como LDA, QDA e Naive Bayes, que modelam o processo de geração dos dados e utilizam o Teorema de Bayes para estimar probabilidades de pertencimento às classes. Por fim, destacamos as vantagens, limitações e pressupostos de cada abordagem, reforçando a importância de escolher o método mais adequado ao contexto do problema.

Na aula de hoje

## Do Número para a Língua, da Língua para o Número

Na aula anterior, discutimos como aplicar classificadores (modelos de aprendizado supervisionado para classificação) no contexto de variáveis quantitativas numéricas: tinhamos características quantitativas de indivíduos (renda, saldo do cartão, etc.) e queríamos classificar esses indivíduos como potenciais devedores ou não; Ou tínhamos informações de saúde (também quantitativas) e queríamos classificar os indivíduos como pessoas com diabetes ou não. No entanto, o objetivo do curso é ensiná-los a classificar observações em categorias (favorável/desfavorável, positivo/negativo) de acordo com o conteúdo textual (conteúdo de uma publicação no *Twitter*). Para isso, precisamos entender como representar a língua de maneira computacional. Daí, surge a seguinte pergunta: Qual a unidade mínima quando tratamos, computacionalmente, a língua? Para Caseli e Nunes (2024, {cite}`caseli_nunes_pln_2024`.), isso depende do seu critério e finalidade: Na fonologia, será o fonema; na morfologia, o morfema; podemos separar a língua em palavras, caractéres, e assim por diante. Para os propósitos dessa disciplina, focaremos em uma unidade: O *Token*.



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Ao fazer o processamento computacional de textos escritos, a definição de que tipo de unidade de processamento se quer buscar/estudar parece estar atrelada às necessidades da tarefa ou trabalho pretendidos. Geralmente, considera-se que uma palavra é, simplesmente, uma unidade grafológica delimitada, nas línguas europeias, entre espaços em branco na representação gráfica, ou entre um espaço em branco e um sinal de pontuação. Essa é uma definição bastante concreta, e bastante prática. No entanto, ao pensarmos em nossos modelos computacionais e suas aplicações no mundo, é importante nos aprofundarmos um pouco mais na conceituação do que é uma palavra e nas possibilidades de processamento e implicações das decisões tomadas no pré-processamento dos corpora."
({cite}`caseli_nunes_pln_2024`., p. 68, tradução nossa)
```


### Token e Type

*Token* é um termo que significa qualquer sequência de caractéres à qual se atribui um valor. Nas línguas européias, a sequência consiste em caractéres delimitados por espaços gráficos, e a tokenização é ajustada para lidar com sinais de pontuação. No entanto, isso não é verdade para todas as línguas. Mas, com essa definição, podemos associar o *token* à palavra escrita. E o *type* seriam os tokens/palavras únicos encontrados em uma frase ou texto. Vamos retomar o que foi discutido na seção "*Bag-of-words* e o modelo multinomial" da terceira aula do curso. Vimos que o modelo *bag-of-words* tem como ideia principal a de representar cada documento pelo número de vezes que cada palavra aparece nele. No exemplo, tínhamos:

1. O cachorro ama o osso;
2. O Osso ama o cachorro;

Ignorando o artigo "O", teríamos a seguinte matriz *Document-feature*, ou o seguinte BOW (*Bag-of-words*):


<div align="center">

| Documento | Cachorro | ama | Osso |
|-----------|------|-------|----------|
| Doc1      | 1    | 1     | 1        |
| Doc2      | 1    | 1     | 1        |

</div>

Essa matriz é chamada assim por que cada linha (ou observação) contém um documento/sentença/frase e cada coluna contém a palavra em consideração. Por exemplo, a coluna "Cachorro" mostra quantas vezes o termo Cachorro aparece em cada documento. Neste caso, os dois documentos possuem as mesmas palavras, e seriam tratados como "iguais". Isso se deve ao fato de que, nesse tipo de representação, a ordem das palavras e o contexto não são considerados; por isso, embora as duas frases expressem relações diferentes, a Bag-of-Words as representa de forma idêntica, pois contêm as mesmas palavras com as mesmas contagens. 

"Tokenizar", ou transformar em *tokens* faz com que cada documento (ou observação na matriz *Document-Feature*) será quebrado em suas partes individuais, as palavras. Esse é o primeiro passo para a criação do saco de palavras (BOW). O exemplo mais comum é o do tokenizar em n-grams de 1. A frase "Diga não à vacinação obrigatória!" pode ser quebrada da seguinte forma:


<div align="center">

["Diga","não","à","vacinação","obrigatória", "!"]

</div>


Seguindo a mesma frase e a mesma etapa de tokenização em palavras, podemos formar bi-grams, que são pares de palavras consecutivas no texto. Bi-grams ajudam a capturar um pouco de contexto local que o unigram (n=1) não representa, como expressões fixas, negações e relações imediatas entre termos.

<div align="center">

["Diga_não", "não_à", "à_vacinação", "vacinação_obrigatória", "obrigatória_!"]

</div>

Se o objetivo é capturar resistência à vacinação, ou hesitação vacinal, talvez quebrar o texto em bigramas seja mais informativo do que quebrá-lo apenas em suas palavras individuais. Com isso, se dá mais contexto ao modelo de aprendizado de máquina, ao custo de adicionar mais combinações raras que podem não aparecer tanto em seu banco de dados. 💬 "Usar ordens maiores de n-gramas pode aumentar substancialmente o número de tipos únicos, mas pode ajudar nossa análise textual ao reter mais informações" ({cite}`grimmer2022text`, p. 99, tradução nossa). Novamente, isso não é uma escolha trivial: Assim como todos os passos e princípios discutidos nas últimas aulas, a forma de processar e representar numericamente o texto altera substancialmente os resultados. O pesquisador deve tentar sempre estar consciente dessas escolhas e relatá-las aos leitores quando necessário. 

### Processamento Morfológico

Para desenvolver qualquer aplicação de PLN, é necessário realizar fases/etapas que convencionamos chamar de pré-processamento do texto. No pré-processamento, algumas tarefas usuais são: Segmentação do texto em sentenças (Sentenciação); Separação de Palavras (tokenização); tokenização em subpalavras; normalização de palavras (lematização, radicalização), entre outras. Como as tarefas mais usuais foram discutidas na seção "*Bag-of-words* e o modelo multinomial", não iremos repetir o conteúdo, partindo para os modelos de linguagem.

## Modelos de Linguagem/Língua

Modelos de linguagem são sistemas matemáticos ou computacionais desenvolvidos para representar e analisar padrões presentes em textos, fala ou outras formas de comunicação. Eles buscam capturar as regularidades estatísticas da língua, como a frequência de palavras, a probabilidade de sequências de termos e as relações contextuais entre diferentes elementos do texto. Esses modelos podem ser aplicados tanto à linguagem natural quanto a linguagens formais, como códigos ou expressões matemáticas. Em PLN, modelos de linguagem são fundamentais para tarefas como previsão da próxima palavra, análise de sentimentos, tradução automática e geração de texto, pois permitem transformar a linguagem em representações numéricas que podem ser processadas por algoritmos de aprendizado de máquina.

Um **Modelo de Linguagem** é um modelo de aprendizado de máquina que faz uma previsão sobre as próximas palavras. Formalmente, um modelo de linguagem atribui uma probabilidade para cada próxima palavra possível, podendo atribuir probabilidades para frases inteiras. Por exemplo, um modelo de linguagem pode dizer que a seguinte frase possui alta probabilidade:

- "Do nada, percebi três homens na calçada"

E atribuirá baixa probabilidade para a seguinte frase:

- "Na calçada três nada do percebi homens"

Para que precisaríamos prever a próxima palavra? *LLMs* são construídas só sendo treinadas para prever palavras, e hoje são muito presentes e possuem diversas aplicações possíveis. O modelo mais simples de língua é o ***N-Gram***, que é uma sequência de n palavras. Por exemplo, um bigrama (ou *2-gram*) poderia ser:

1. "A água"
2. "O copo"
3. "A vacina"

E trigramas:

1. "Copo de água"
2. "Vacina da Covid"
3. "Presidente do Brasil"

Mas um *N-gram* também é um modelo de probabilidade[^1] que estima a probabilidade de uma palavra dada as n-1 palavras que vem anteriormente.

### *N-Grams*

Começaremos com a tarefa de estimar a $Pr(p|h)$, a probabilidade da palavra $p$ dado o histórico $h$. Suponha que o histórico $h$ seja "A praia de Copacabana é tão" e queremos saber a probabilidade de que a próxima palavra seja "azul". Portanto, queremos estimar:

$$
Pr(Azul | \text{A praia de Copacabana é tão})
$$

Uma forma de estimar essa probabilidade é por meio da contagem de frequências: Dado um córpus[^2], quantas vezes a frase "A praia de Copacabana é tão" é seguida por "Azul".

$$
 Pr(\text{blue} | \text{A praia de Copacabana é tão}) \;=\;
\frac{C(\text{A praia de Copacabana é tão Azul})}
     {C(\text{A praia de Copacabana é tão})} 
$$

No entanto, nenhum córpus será tão grande a ponto de nos dar boas estimativas para essa probabilidade. Isso se deve ao fato da Língua e a Linguagem serem criativas, e novas frases são criadas o tempo todo. Por isso, outra forma de estimar a probabilidade é necessária. Uma forma de estimar essa probabilidade é por meio da *Chain Rule of Probability* (Ou Regra Geral do Produto/Cadeia, em português). Aplicando ela para palavras ($p$), temos:

$$
Pr(P_1, ..., P_n) = Pr(P_1) P(P_2|P_1) P(X_3|X_{1:2}) ... P(X_n|X_{1:n-1}) 
$$

De forma geral:

$$
\prod_{k=1}^{n} P\bigl(p_k \,\bigl|\, p_{1{:}k-1}\bigr)
$$

Ou seja, podemos estimar a probabilidade conjunta de uma frase inteira por meio da multiplicação das probabilidades condicionais que a compõem. Dito de outra forma, a regra geral do produto diz que podemos calcular a probabilidade de uma frase multiplicando as probabilidades de cada palavra aparecer, considerando as palavras anteriores. Assim, mesmo sem ter todas as frases no nosso banco de dados, conseguimos estimar a chance de uma sequência de palavras acontecer. No entanto, como calcular cada probabilidade condicional (e.g. $Pr(P_2|P_1)$)?

### A Suposição de Markov

A intuição por trás do modelo *N-gram* é de que, ao invés de computar a $Pr(p|h)$, podemos aproximar o histórico $h$ só com as últimas palavras.  O modelo de bigrama, por exemplo, aproxima a probabilidade de uma palavra dada todas as palavras anteriores $Pr(p_n|p_{1:n-1})$ usando a probabilidade condicional da palavra anterior $Pr(p_n|p_{n-1})$. Ou seja, no lugar de estimar


$$
Pr(Azul | \text{A praia de Copacabana é tão})
$$


Ele aproxima $h$ por meio da probabilidade:

$$
Pr(Azul | tão)
$$

De maneira geral, a seguinte aproximação é feita

$$
Pr(p_n|p_{1:n-1}) \approx Pr(p_n|p_{n-1})
$$

Esse pressuposto, ou suposição, de que a probabilidade de uma palavra depende apenas da palavra anteiror é chamado de **Suposição de Markov**. Modelos de Markov são uma classe de modelos probabilísticos que assumem que podemos prever a probabilidade de uma unidade futura sem olhar muito distante no passado. Portanto, a probabilidade de uma frase inteira pode ser estimada por

$$
Pr(p_{1:n}) \approx \prod_{k=1}^{n} Pr(p_k|p_{k-1})
$$

### Como estimar as probabilidades?

Uma forma de estimar essas probabilidades que utilizam a suposição de Markov é chamada de *Maximum Likelihood Estimation*, ou **Método de Estimação de Máxima Verossimilhança**. Em texto, conseguiremos as estimativas em um modelo *n-gram* pegando contagens de um córpus, e normalizando[^3] essas contagens para que fiquem entre 0 e 1. Por exemplo, para computar a probabilidade de um bigrama de uma palavra $p_n$ dada uma palavra anterior $p_{n-1}$, se computa a contagem de um bigrama $C(p_{n-1} p_{n})$ e normalizar essa contagem pela soma de todo os bigramas que compartilham a primeira palavra $p_{n-1}$:


$$
P\bigl(p_n \,\bigl|\, p_{n-1}\bigr)
   \;=\;
   \frac{C\!\bigl(p_{n-1}p_n\bigr)}
        {\displaystyle\sum_{p} C\!\bigl(p_{n-1}p\bigr)}
$$

Vamos trabalhar alguns exemplos de cálculos de probabilidade só para entender como funciona (<f> e </f> indicam o começo e o fim de uma frase):
 
$$
<f> \text{Eu sou João} </f>
$$

$$
<f> \text{João sou eu} </f>
$$

$$
<f> \text{Eu não gosto de sopa} </f>
$$


Se esse fosse nosso córpus inteiro, poderíamos computar as seguinte probabilidades para os bigramas:

$$
Pr (Eu | <f>) = 2/3 \approx 67%
$$

Aqui, queremos a probabilidade de que a frase começe com "Eu". Em dois casos (1 e 3 frases), isso ocorre. Obtemos, então, uma probabilidade de aproximadamente 67%. Para a frase abaixo

$$
Pr (João | <f>) = 1/3
$$

Queremos a probabilidade de que a frase começe com "João". Só uma frase das três começa, portanto a probabilidade é de 1/3, ou aproximadamente $33\%$. No caso geral, a estimação de paramêtros no *MLE* (Maximum Likelihood Estimation) em n-grams fica:

$$
P\bigl(w_n \,\bigl|\, w_{\,n-N+1{:}n-1}\bigr)
   \;=\;
   \frac{C\!\bigl(w_{\,n-N+1{:}n-1}\,w_n\bigr)}
        {C\!\bigl(w_{\,n-N+1{:}n-1}\bigr)}
$$

Os modelos probabilísticos, como os n-grams, permitem calcular a probabilidade de diferentes sequências de palavras em uma língua, atribuindo valores a cada possível continuação de uma frase com base nas ocorrências observadas em um córpus. Esse cálculo é fundamental para tarefas de geração automática de texto, pois possibilita escolher, a cada etapa, a palavra mais provável para continuar a sentença. O processo de busca gulosa utiliza exatamente essa ideia: a cada passo da geração, seleciona a palavra com maior probabilidade condicional, formando sentenças que refletem os padrões estatísticos aprendidos pelo modelo. Assim, a busca gulosa é uma estratégia prática que conecta diretamente os cálculos probabilísticos dos modelos de linguagem à produção de frases coerentes e naturais. A {numref}`Figura {number} <figgulosa>` ilustra o processo de geração de sentenças em um processo de "Busca Gulosa", um dos possíveis dentre vários.


```{figure} ../aula7/images/fig17.1.png
---
width: 100%
name: figgulosa
align: center
---
Exemplo de geração de sentença em um processo de "Busca Gulosa". Fonte Caseli e Nunes (p.369, {cite}`caseli_nunes_pln_2024`.)
```

Compreender os modelos de linguagem n-gram e os modelos probabilísticos apresentados nesta aula é fundamental para realizar tarefas de classificação com texto em Processamento de Linguagem Natural. Esses modelos permitem transformar textos em representações numéricas que capturam padrões de frequência, contexto e dependência entre palavras, tornando possível aplicar algoritmos de aprendizado de máquina para identificar categorias, sentimentos ou tópicos em documentos.

O modelo n-gram, ao considerar sequências de palavras, vai além da simples contagem individual de termos, incorporando informações sobre o contexto imediato e relações locais entre palavras. Isso é especialmente útil para distinguir nuances de significado, identificar expressões fixas e melhorar a precisão dos classificadores. Já os modelos probabilísticos, ao estimar a probabilidade de ocorrência de sequências de palavras, fornecem uma base estatística sólida para a tomada de decisão em tarefas de classificação, seja para prever a próxima palavra, identificar o sentimento de um texto ou categorizar documentos.

Ao dominar esses conceitos, o pesquisador consegue construir representações mais informativas dos textos, escolher as melhores estratégias de pré-processamento e selecionar modelos adequados para diferentes problemas de classificação. Dessa forma, o entendimento dos modelos de linguagem n-gram e probabilísticos é um passo essencial para o desenvolvimento de soluções eficazes e interpretáveis em PLN.

### Próximos Passos e Modelos Neurais

Os modelos probabilísticos de linguagem, como os n-grams, inauguraram a ideia de atribuir uma probabilidade explícita a cada sequência de palavras por meio da regra da cadeia e da suposição de Markov. Esse enquadramento mostrou que textos podiam ser convertidos em contagens normalizadas e treinados por máxima verossimilhança, introduzindo métricas como perplexidade para avaliar a qualidade do modelo. Quando surgiram os modelos neurais, eles mantiveram o mesmo objetivo de estimar $Pr(p|h)$, mas trocaram tabelas esparsas por vetores densos e parâmetros aprendidos, conseguindo generalizar para contextos mais longos e lidar melhor com dados raros. Entender essa herança probabilística nos faz perceber que mesmo as arquiteturas mais recentes continuam, no essencial, sendo máquinas de previsão de sequências. Um princípio que permanece nos grandes modelos de linguagem atuais, como os *LLMs*.



## Regressão Logística e Classificação com Texto

Discutimos ao longo do curso que a tarefa de classificação envolve pegar preditores $X$ (ou *features*) e utilizá-los para tentar prever à que categoria $Y$, nosso *target* pertence. Para a classificação com texto, a principal mudança é a de que não utilizaremos mais dados numéricos (Saldo do cartão, variáveis de saúde), e sim **Texto**, pré-processado e representado numericamente. Um exemplo clássico dentro da classificação com texto é a de verificar se uma avaliação de um produto é positiva ou negativa. Isso é uma tarefa dentro da área de Análise de Sentimentos, e as categorias podem variar um pouco. Nessa tarefa, pegamos o texto da avaliação {"O produto é muito bom!", "Não gostei nem um pouco"} e tentaremos classificar eles como positivos ou negativos. Como vimos antes, um classificador básico é a regressão logística, e ela servirá de base para entendermos o mecanismo por trás dos modelos de aprendizado supervisionado com texto.


### Regressão Logística

No ISL ({cite}`james2023introduction`.), os autores explicam a Regressão Logística por uma linguamge mais estatística. Já Jurafsky e Martin ({cite}`jurafsky2024speech`.) vão por uma linha mais do Aprendizado de Máquina. Por isso, não estranhem a mudança de jargão e de termos para se referir à Regressão Logística. 

#### Função Sigmóide

Como dito antes, o objetivo da regressão logística binária é treinar um classificador capaz de tomar uma decisão binária sobre a classe de uma nova observação de entrada. Aqui introduzimos o classificador sigmoide, que nos ajudará a tomar essa decisão. Em Jurafsky e Martin, a Regressão logística resolve o problema de estimar $Pr(Y=1|X)$ (A probabilidade de que a observação pertence à classe 1, dado os preditores) estimando um vetor de **pesos** e **termos de viés**. Vamos chamar os pesos de $w$ e o viés de $b$. Cada peso $w_i$ é um número real e está associado à um dos preditores $X_i$. O peso representa quão importante é cada preditor para a decisão de classificação, podendo ser negativo ou positivo. Em uma tarefa de classificação de sentimento, provávelmente a palavra "ótimo" terá um peso positivo, e a palavra "horrível" terá um peso negativo. O termo de viés, ou intercepto, é um número real que é adicionado aos *inputs* ao final do cálculo. Para fazer uma decisão em uma observação de teste (após o treinamento), o classificador logístico vai multiplicar cada preditor $X_i$ pelo seu peso $w_i$, e somar isso com o termo de viés. Formalmente, temos:

$$
z \;=\; \left(\sum_{i=1}^{n} w_i x_i\right) + b
$$

No entanto, os valores de $z$ resultantes dessa equação não estão obrigatoriamente entre 0 e 1. Para que possamos calcular a probabilidade de que uma observação é da classe 0 ou 1, precisamos limitar z para esse intervalo. Para isso, passaremos z pela função sigmóide $\sigma$, ou função logística.

$$
\sigma(z)
   \;=\;
   \frac{1}{1 + e^{-z}}
   \;=\;
   \frac{1}{1 + \exp(-z)}
$$







## Notas

[^1]: O termo **n-gram** pode ser usado em dois sentidos: (1) como uma sequência de n itens (palavras, caracteres, etc.) extraída de um texto, e (2) como um modelo de linguagem que estima a probabilidade de uma palavra ou sequência com base nas n-1 palavras anteriores. O contexto geralmente indica qual sentido está sendo utilizado.

[^2]: O termo **córpus** refere-se a um conjunto estruturado de textos ou documentos utilizados para análise linguística ou treinamento de modelos de linguagem. Em PLN, o córpus serve como fonte de dados para extrair padrões, calcular frequências e estimar probabilidades, sendo fundamental para o desenvolvimento e avaliação de métodos computacionais aplicados à linguagem. Aqui, estamos indo para além da ideia de um córpus anotado.

[^3]: **Normalizar** significa ajustar os valores de uma variável ou conjunto de dados para que fiquem dentro de um intervalo padrão, geralmente entre 0 e 1. No contexto de modelos de linguagem, normalizar as contagens transforma frequências absolutas em probabilidades, facilitando a comparação e o processamento estatístico dos dados.

