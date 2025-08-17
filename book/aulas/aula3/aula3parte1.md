
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
| Doc2      | 0    | 1     | 1         |

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

Depois de todos esses passos, o final é o de transformar o texto em dados numéricos. O resultado final é uma matriz de documentos-termos, chamada de *W*, com N textos e um vocabulário de tamanho J (Matriz N x J). No geral, essa matriz é muito esparsa, isto é, contém muitos zeros em muitas colunas para muitos documentos. Por isso é necessário tokenizar, eliminar certos termos, etc. 


### Quando não pré-processar?

Nem sempre vale aplicar todo o pré-processamento “no automático”. Em tarefas que dependem de nuances de forma ou contexto — como reconhecimento de entidades (nomes próprios, marcas, locais), análise estilística, detecção de ironia/sarcasmo, distinções semânticas sensíveis à capitalização (“Porto” vs. “porto”), ou quando a pontuação e emojis carregam sinal (sentimento, ênfase) — remover pontuação, fazer lowercasing agressivo, tirar stopwords ou reduzir à raiz pode apagar informação útil. Além disso, em domínios especializados (jurídico, biomédico), a forma exata e a morfologia costumam importar; já em textos curtos (como títulos e tweets), cada token tende a ter mais peso, e filtros pesados podem “esvaziar” o sinal. Por isso, é recomendável: alinhar cada transformação ao objetivo da tarefa; testar incrementalmente o impacto de cada etapa; manter versões reproduzíveis do pipeline; ajustar listas de stopwords ao domínio; preferir lematização a stemming quando a precisão lexical for crucial; registrar decisões e métricas; e, quando fizer sentido, combinar representações (p.ex., manter capitalização como feature adicional, preservar certos sinais de pontuação/emoji ou incluir n-grams) em vez de eliminar informação de forma irreversível.



## Capítulo 6 - O Modelo Multinomial

Até agora, vimos como transformar um texto em números usando *bag‑of‑words* (BOW).  
Mas uma pergunta importante surge: **de onde vêm esses números?**  
Ou, em termos de estatística: *qual a "história" por trás das contagens de palavras?*

Neste capítulo, Grimmer et al. discutem a **distribuição multinomial**, que é o modelo probabilístico mais simples e direto para textos representados como BOW.  
Esse modelo permite **atribuir probabilidades a documentos** e pensar em hipóteses, como por exemplo: *quem é o autor de um texto?*  

Aqui é necessário distinguir entre modelos probabilísticos e modelos algorítmicos, que são duas formas diferentes de ver a mesma coisa. Modelos probabílisticos olham para como os dados foram gerados usando probabilidade. Quando as premissas sobre o processo de geração dos dados (Ou *Data Generating Proces*, *DGP*) são corretas, os modelos probabilísticos oferecem uma afirmação clara sobre as premissas no funcionamento ideal do modelo. Também são fáceis de extrapolar para novas situações e apresentam formas claras para otimização e quantificação da incerteza.

```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Essas bases probabilísticas contrastam com abordagens algorítmicas, que especificam uma série de passos, geralmente na forma de uma função objetivo a ser otimizada. É melhor pensar nisso como uma linguagem diferente para descrever algo semelhante. Por exemplo, muitos modelos, como a regressão linear, podem ser descritos de uma perspectiva probabilística (um modelo linear normal) ou de uma perspectiva algorítmica (minimizar a soma dos erros quadráticos)."
({cite}`grimmer2022text`, p. 112, tradução nossa)
```

Em estatística e em ciência de dados, *Data Generating Process (DGP)*, ou processo gerador de dados, é a narrativa formal — probabilística ou causal — que especifica como os dados observados poderiam ter sido produzidos. Em outras palavras, é um conjunto de suposições explícitas sobre:

* Quais variáveis existem (observadas e latentes).

* Como elas se relacionam (determinística e/ou estocasticamente).

* Quais distribuições de probabilidade regem os componentes aleatórios.

* Como o “ruído” entra no sistema e quais são suas propriedades (média, variância, independência/heterocedasticidade, etc.).

* Qual é a ordem de geração (quem vem antes: parâmetros, covariáveis, resultados).

A utilidade do DGP é dupla. Primeiro, ele torna transparentes as suposições do modelo: quando se sabe exatamente que história está sendo contada, sabe-se também quando as conclusões são válidas. Segundo, um DGP bem especificado habilita a escolha de estimadores apropriados, a derivação de propriedades (consistência, viés, variância), a quantificação de incerteza e a validação do modelo (diagnósticos e testes de aderência).

Em modelos probabilísticos, o DGP é escrito como uma sequência de sorteios de distribuições (por exemplo, parâmetros são sorteados de um prior; dados são sorteados de uma verossimilhança condicional nos parâmetros). Isso permite inferência Bayesiana, estimação por máxima verossimilhança e análise de incerteza bem fundamentada. Em abordagens algorítmicas, muitas vezes não se escreve explicitamente um DGP; define-se um objetivo de otimização (loss) e um procedimento de ajuste. Mesmo assim, pode-se interpretar esses procedimentos como aproximando um DGP implícito (por exemplo, regressão linear com MSE corresponde a erros normais i.i.d.).


### Revisão — Distribuições, DGP e Regressão Linear

Modelar estatisticamente é **contar uma história** sobre como os dados surgem. Chamamos essa história de **processo gerador de dados (DGP)**.  
No caso mais simples:

1. **Distribuição** escolhida – Normal, Bernoulli, Multinomial, etc.  
2. **Parâmetros** que controlam média, variância, covariâncias.  
3. **Regras de amostragem** (independência, tamanho da amostra).  

Quando alinhamos essas peças, obtemos previsões (médias, variâncias) que podem ser comparadas à observação.

---

#### Regressão Linear Simples


$$
\underbrace{Y_i}_{\text{quantidade observada}}
=\;\beta_0+\beta_1 X_i \;+\;\underbrace{\varepsilon_i}_{\text{ruído}}
,\qquad
\varepsilon_i\sim\mathcal{N}\bigl(0,\sigma^2\bigr)
$$

* **Distribuição** dos erros: Normal(0, σ²).  
* **Parâmetros desconhecidos**: $\beta_0, \beta_1, \sigma^2$.
* **Variáveis explicativas**: $X_i$  
* **Suposição de independência** entre $\varepsilon_i$.  

Isso gera um conjunto de $Y_i$ cujas **condicionais** $Y|X$ seguem uma normal centrada em $\beta_0+\beta_1X$. Por que assumir erro Normal?

**A. Lembrete-relâmpago: Distribuição Normal**  
• Formato “sino”: simétrica em torno da média $\mu$.  
• Desvio-padrão $\sigma$ controla a “largura” (68 % dos valores em $\mu \pm \sigma$).  
• Única distribuição contínua totalmente descrita por **média** e **variância**.

**B. Teorema Central do Limite (TCL) em 2 linhas**  
Se somarmos (ou tirarmos a média de) muitos efeitos independentes com variância finita, o resultado tende a ser Normal — **mesmo que cada efeito individual não seja Normal**.  
Consequência prática: o termo de erro $\varepsilon$ de um modelo costuma ser bem aproximado por Normal, porque ele agrega inúmeras pequenas influências não controladas.


#### Conexão com o Multinomial (Cap. 6 de Grimmer et al.)

No capítulo de texto como dado:

* **Distribuição de contagem** escolhida: **Multinomial**.  
* **Parâmetro**: vetor $\mu$ no simplex (probabilidades das palavras).  
* **Dados**: contagens $W_{ij}$.  
* **DGP**: para cada documento i  
  $$
  W_i \sim \text{Multinomial}(M_i,\mu).
  $$

A lógica é idêntica à regressão:

1. Escolher a distribuição coerente com a natureza do dado (contagem → Multinomial).  
2. Declarar suposições (independência de tokens, mesmo $\mu$ em todos os docs ou dentro de grupos).  
3. Estimar $\mu$ pela máxima verossimilhança ($\hat\mu_j=W_{\cdot j}/\sum_j W_{\cdot j}$).  
4. Avaliar ajuste e, se necessário, **regularizar** com um prior Dirichlet (equivalente a adicionar pseudo-contagens).

---

#### Papel das suposições — paralelos úteis

| Componente            | Regressão linear               | Multinomial de texto          |
|-----------------------|--------------------------------|-------------------------------|
| Variável de interesse | Y contínuo                     | Vetor de contagens            |
| Distribuição do erro  | Normal                         | Multinomial                   |
| Independência         | Observações \(i\)              | Tokens dentro de doc          |
| Heterocedasticidade?  | Viola σ² constante             | Viola $\mu$ comum           |
| Regularização         | Erros-padrão robustos, Bayes ridge | Dirichlet prior (\(\alpha\)) |

Quebrar qualquer suposição exige **diagnóstico** (gráficos de resíduos, comparação empírica/vistas teóricas) e possivelmente **refinar o DGP** (transformar variáveis, hierarquizar parâmetros, robustez).



### Distribuição Multinomial

Imagine que temos só **três palavras possíveis no vocabulário**: gato, cachorro e peixe.  
Cada documento é um "saco de palavras" com algumas dessas palavras.  

Se o documento tiver **apenas uma palavra**, podemos representá-lo assim:

- gato → $(1, 0, 0)$  
- cachorro → $(0, 1, 0)$  
- peixe → $(0, 0, 1)$  

Isso é chamado de representação *one‑hot encoding* (só um valor igual a 1, o resto é 0). Também é conhecido como dummy variable, com cada coluna tendo valores binários (zero e um) para cada palavra.

---

#### Ligação com a probabilidade

Podemos imaginar um **sorteio de palavras**: para cada posição do documento, escolhemos uma palavra de acordo com certas probabilidades.  

Por exemplo, suponha que temos:

$$
\mu = (0.5,\, 0.25,\, 0.25)
$$

Isso significa que:
- metade das vezes sai **gato**,
- em 25% das vezes sai **cachorro**,
- e em 25% das vezes sai **peixe**.

---

#### Quando o documento tem mais de uma palavra

Agora, em vez de escolher uma palavra, escolhemos várias (por exemplo $M=10$ palavras).  
O documento é uma **coleção de sorteios**. O modelo que descreve isso é a **distribuição multinomial**.

Formalmente, se $W_{ij}$ é a contagem da palavra $j$ no documento $i$, dizemos:

$$
W_i \sim \text{Multinomial}(M_i, \mu).
$$

E a fórmula da probabilidade desse documento é:

$$
p(W_i \mid \mu) = \frac{M_i!}{\prod_{j=1}^J W_{ij}!}
\prod_{j=1}^J \mu_j^{W_{ij}}.
$$

💡 Não se assuste:  
- a fração com fatoriais ($M_i! / \prod W_{ij} !$) só diz **quantas formas diferentes há de reorganizar as palavras dentro do documento**;  
- o produto $\prod \mu_j^{W_{ij}}$ só diz: "qual é a probabilidade de ter exatamente tantas ocorrências de cada palavra".

---

#### Exemplo intuitivo

Diga que $\mu = (0.5, 0.25, 0.25)$, ou seja, metade das vezes dá **gato**.  

Agora criamos um documento com 3 palavras: (peixe, gato, peixe).  
Isso equivale ao vetor:

$$
W = (1, 0, 2).
$$

A probabilidade desse documento é:

$$
p(W \mid \mu) = \frac{3!}{1!\,0!\,2!}(0.5)^1 (0.25)^0 (0.25)^2 = 0.09375.
$$

Ou seja, **9,4% de chance** de observar esse documento, dado o modelo.  

---

#### O que essa distribuição garante?

- **Média**  
  \[
    \mathbb{E}[W_{ij}] \;=\; M_i \,\mu_j
  \]  
  A contagem esperada do termo \(j\) no documento \(i\) é a sua probabilidade \(\mu_j\) multiplicada pelo tamanho do documento \(M_i\).

- **Variância**  
  \[
    \operatorname{Var}(W_{ij}) \;=\; M_i \,\mu_j \,(1-\mu_j)
  \]  
  A incerteza é maior quando \(\mu_j \approx 0{,}5\) e diminui quando a palavra é muito rara ou muito comum.

- **Covariância**  
  \[
    \operatorname{Cov}\!\bigl(W_{ij},\,W_{ij'}\bigr) \;=\; -\,M_i \,\mu_j \,\mu_{j'}
  \]  
  Como o total de tokens é fixo (\(M_i\)), um aumento na contagem de \(j\) implica redução esperada em \(j'\) — efeito de “soma constante”.


---

#### Um Modelo de Linguagem Básico

Um **modelo de linguagem** é qualquer modelo que atribui probabilidades a textos.  
No caso multinomial, pensamos o texto como "gerado" a partir de uma **caixa de probabilidades $\mu$**.

---

#### Aplicação famosa – *Federalist Papers*

Mosteller e Wallace (1963) tentaram descobrir quem escreveu alguns ensaios com autoria disputada.

Considere o vocabulário bem pequeno: {**by**, **man**, **upon**}.  
Contagens observadas:

| Autor     | by  | man | upon |
|-----------|-----|-----|------|
| Hamilton  | 859 | 102 | 374  |
| Jay       | 82  |   0 |   1  |
| Madison   | 474 |  17 |   7  |
| Disputado | 15  |   2 |   0  |

---

#### Estimando $\mu$

Para cada autor, calculamos a *fração de uso* de cada palavra.  
Por exemplo, Hamilton:

$$
\hat{\mu}_H = \left(\tfrac{859}{1335}, \tfrac{102}{1335}, \tfrac{374}{1335}\right)
= (0.64,\, 0.08,\, 0.28).
$$

Fazendo o mesmo para os outros autores, temos:

- Jay: $(0.99, 0, 0.01)$  
- Madison: $(0.95, 0.035, 0.015)$

---

#### Testando o documento disputado

Documento disputado = $(15, 2, 0)$.  

Calculamos a probabilidade de ele surgir segundo cada autor.  
O resultado indica que **Madison** é o autor mais provável.

---

#### Por que precisamos de *Smoothing*?

Note como Jay tem probabilidade **zero** de usar "man".  
Mas será que isso é verdade? Não, só temos poucos textos de Jay. Ele *poderia* ter usado "man", só não apareceu.  

Se aplicarmos as fórmulas “secas”, então qualquer vez que "man" apareça, Jay é automaticamente impossível.  
Isso é perigoso!

---

#### Solução: Suavização de Laplace (add-one)

Em vez de assumir que zero é zero, adicionamos **1 a todas as contagens**:

$$
\tilde{W}_{ij} = W_{ij} + 1.
$$

Assim, toda palavra tem chance não-nula.  
É como imaginar que vimos cada palavra pelo menos uma vez.

---

#### O Dirichlet como Regularização

Podemos ser ainda mais elegantes: em vez de *forçar um add‑one manual*,  
colocamos um **prior probabilístico** sobre $\mu$: a **Distribuição de Dirichlet**.

---

#### O que é uma Dirichlet?

Uma distribuição que gera vetores de probabilidades $\mu$ (não negativos, que somam 1).  
Ou seja, é perfeita para modelar "proporções de palavras".

Se:

$$
\mu_k \sim \text{Dirichlet}(\alpha),
$$

então o vetor $\mu_k$ já vem com uma noção de "suavização".  
Os parâmetros $\alpha_j$ funcionam como **pseudo-contagens**.

Exemplo:  
- Se $\alpha = (1,1,1)$, é como se tivéssemos visto *1 ocorrência fictícia de cada palavra*.  
- Se $\alpha = (10,10,10)$, todos os $\mu$ ficarão próximos de $(1/3,1/3,1/3)$.

---

#### Propriedades

- Esperança:  
  $$ \mathbb{E}[\mu_j] = \frac{\alpha_j}{\sum \alpha} $$
- Variância:  
  valores grandes de $\alpha$ → distribuições concentradas;  
  valores pequenos → maior variabilidade.

---

### 6.5 Conclusão

- A **distribuição multinomial** representa **contagens de palavras**, assumindo sorteios independentes.  
- Ela explica como calcular probabilidades de documentos e possibilita análises de autoria, classificação etc.  
- O problema dos zeros é resolvido com **suavização (Laplace)** ou com um **prior Dirichlet**.  
- Esse é um dos pontos de partida para modelos de tópicos, classificação supervisionada e outras técnicas avançadas.

---

Resumindo: o modelo multinomial é como **imaginar que um autor tem um saquinho de palavras com certas probabilidades, e cada documento é produzido sorteando delas várias vezes**.







