# Aprendizado Supervisionado II

Na última aula, vimos que o Aprendizado de máquina é um campo dentro da Inteligência artificial que busca, com base em experiências prévias (i.e. dados de treinamento), fazer classificações ou previsões para nossa variável de interesse (*target*). Para isso, os métodos de aprendizado estatístico precisam de uma função erro, que busca reduzir a distância entre os valores previstos e os valores reais dos dados de treinamento. Vimos também os passos principais para a criação de um banco de dados de treinamento (Codebook, anotação, concordância entre anotadores), e o *pipeline* básico de uma aplicação de classificação. Destacamos métricas essenciais como acurácia, precisão, recall e F1-score para avaliar o desempenho dos classificadores. Por fim, reforçamos a importância de testar o modelo em dados novos para garantir sua capacidade de generalização e utilidade prática. Na aula de hoje, retomaremos algumas discussões da aula anterior, e discutiremos a diferença entre inferência e predição, métodos paramétricos e não-paramétricos, e os trade-offs clássicos de aprendizado de máquina entre Flexibilidade x Interpretabilidade, e viés x variância.


Como dito, o objetivo de uma tarefa de aprendizado de máquina é usar um conjunto de dados para fazer previsões e classificações para outros dados não observados durante o treinamento. Esse conjunto de dados é conhecido como o **banco de treinamento** (ou Córpus anotado, em PLN), e é composto de dois tipos de variáveis principais: A variável ***Target*** (alvo), representada muitas vezes por $Y$, e também conhecida por variável resposta, variável dependente, ou variável explicada. O segundo tipo de variável é o que é chamado, no *ML*, de ***Features***, representadas por $\mathbf{X}$, que são vetores de variáveis preditoras, ou independentes/explicativas, ou variáveis *input*. No contexto de Processamento da Língua Natural, nosso principal $X$ é o texto em cada documento, e o principal $Y$ (em classificação) são as categorias de interesse, seja sentimento, posicionamento, relevância, tópicos, etc. Na literatura de Aprendizado Estatístico e de Aprendizado de máquina, $X$ e $Y$ podem ser referidos de todas as formas mencionadas acima, mas significam a mesma coisa. 

Com esse banco de dados de treinamento, contendo $\mathbf{X}$ e $Y$, o principal objetivo é de modelar a relação entre eles, tal que tenhamos

$$
 Y = f(x) + \epsilon
$$

Em que $Y$ é nosso *target* (e.g. Sentimento, posicionamento) e $\mathbf{x}$ é o vetor de nossas variáveis explicativas (o Texto em representação numérica). $f(x)$ seria, portanto, uma função geral que representa a relação entre nossas *features* e o *target*.

## Por que estimar $f(x)$?


Existem dois contextos em que o pesquisador está interessado em estimar $f(x)$: Inferência e Predição. Grosso modo, as pesquisas em **inferência** procuram entender o impacto de cada variável explicativa ($X_1, X_2, ..., X_p$) em $Y$, e como essa relação se altera com a inclusão de novas variáveis, interações, polinômios, etc. Por exemplo, uma pesquisa pode estar preocupada em entender como a religião de um indivíduo pode impactar em seu apoio ao bolsonarismo. Um possível resultado dessa pesquisa poderia ser de que o indivíduo ser evangélico tem um efeito positivo constante no apoio ao bolsonarismo, em comparação com outras religiões/denominações. 

No contexto da **predição**, o foco é em utilizar os dados de treinamento rotulados (e, com isso, as variáveis $X_1, X_2, ..., X_p $) para prever os valores de $Y$, sejam estes valores contínuos ou categóricos. Um exemplo clássico de classificação neste contexto é o de classificação de e-mails em *Spam* ou não *Spam*. Nessa tarefa, utiliza-se o texto do email em alguma representação numérica (*Bag-of-words*, *embeddings*, etc.) para a classificação binária de *Spam* ou não. 

### Predição

No contexto de predição, estimamos os seguintes valores:

$$
\hat{y} = \hat{f}(\mathbf{x})
$$


Onde $\hat{f}$ é a estimativa de $f$ e $\hat{y}$ é a estimativa de $y$. Aqui, $\hat{f}$ é tratado como uma "caixa preta", no sentido de que a preocupação não é com sua forma, nem com sua especificação, mas se ele fornece boas previsões de $y$. No exemplo do e-mail, não importa quais palavras são melhores preditoras de se um e-mail é ou não *spam*, mas sim que o modelo consiga classificar corretamente essa categoria, na maior parte dos casos. A precisão da função $\hat{f}$ é determinada por dois tipos de erro: um redutível e outro irredutível. Mesmo que tenhamos um ótimo modelo e especificação, ainda existirá uma parcela de erro devido à fatores estocásticos (i.e. aleatórios).


````{margin}
```{note}
Um problema comum que pode existir em aplicações de aprendizado de máquina é o *data leakage*. *Data leakage* é um problema que ocorre quando informações do conjunto de teste ou de validação acabam sendo utilizadas, direta ou indiretamente, durante o treinamento do modelo. Isso faz com que o modelo tenha acesso a dados que não deveria conhecer, levando a resultados artificialmente altos nas métricas de avaliação e prejudicando sua capacidade de generalização para dados realmente novos. Portanto, sabendo que existe um erro irredutível nas aplicações de predição, resultados **bons demais** na validação e teste (i.e. resultados muito próximos da perfeição) podem indicar que o pesquisador está com vazamento de dados.
```
````

$$
 E(y-\hat{y}) = E[f(x)+ \epsilon - \hat{f}(x)]² \\
 = [f(x)-\hat{f}(x)]² + Var(\epsilon)
$$

Onde

$$
\underbrace{[f(x)-\hat{f}(x)]^2}_{\text{Erro redutível: diferença entre a função verdadeira e a estimada}} \\
$$

$f(x)$ seria a verdadeira relação de variáveis que melhor explicam e prevem $y$ (ou o verdadeiro *Data Generating Process*) e $\hat{f}(x)$ é a função que o pesquisador estabeleceu com as variáveis existentes no banco de dados. Sempre é possível, com base na rotulação de treinamento, reduzir a diferença entre o que encontramos nos dados e o que melhor aproxima $y$. No entanto, o outro componente da equação é


$$
+ \underbrace{Var(\epsilon)}_{\text{Erro irredutível: variabilidade aleatória dos dados}}
$$

Esse erro é irredutível e estocástico, e sempre estará presente em qualquer aplicação, seja ela inferencial ou de previsão. Esse erro faz com que, independente da nossa especificação de $\hat{f}(x)$, $E(y-\hat{y})$ nunca será igual a zero. Aqui estão dois exemplos de tarefas de predição:

**Predição de valores contínuos (Regressão):**  
Um pesquisador deseja prever o preço de casas em uma cidade com base em variáveis como número de quartos, área construída, localização e idade do imóvel. O banco de treinamento contém registros dessas características ($\mathbf{X}$) e o preço real de cada casa ($Y$). O objetivo é estimar uma função $\hat{f}(\mathbf{x})$ que, ao receber as características de uma nova casa, forneça uma previsão do seu preço ($\hat{y}$), um valor contínuo. 

**Predição em PLN com classificação:**  
Em Processamento de Linguagem Natural, uma tarefa comum é classificar textos em categorias específicas. Por exemplo, considere um sistema de análise de sentimentos aplicado a avaliações de produtos online. O banco de treinamento é composto por textos de avaliações ($\mathbf{X}$) e o rótulo correspondente ($Y$), indicando se a avaliação é positiva, negativa ou neutra. O modelo aprende padrões linguísticos e de frequência de palavras para estimar $\hat{f}(\mathbf{x})$ e, ao receber uma nova avaliação, prevê o sentimento expresso pelo usuário ($\hat{y}$), realizando uma classificação multiclasse. Esse tipo de tarefa é essencial para empresas que desejam monitorar a satisfação dos clientes, identificar problemas recorrentes em produtos ou serviços, e tomar decisões estratégicas baseadas no feedback dos usuários.

Outro exemplo relevante de classificação em PLN é a detecção automática de notícias falsas (*fake news*). Nesse caso, o banco de treinamento contém textos de notícias ($\mathbf{X}$) e o rótulo ($Y$) indicando se a notícia é verdadeira ou falsa. O modelo pode ser treinado para identificar padrões de linguagem, fontes e estrutura textual que diferenciam notícias confiáveis de notícias enganosas, auxiliando plataformas digitais e leitores na filtragem de informações e combate à desinformação.

### Inferência

No contexto da inferência, também há a preocupação de estimar $f$. No entanto, o foco está em entender a associação entre $y$ e $X = \{X_1, X_2, ..., X_p\}$. Diferente da predição, onde o objetivo principal é prever valores futuros ou desconhecidos, a inferência busca interpretar e explicar como as variáveis explicativas influenciam o resultado. Algumas perguntas comuns nesse tipo de estudo incluem:

- Quais variáveis explicativas estão associadas com $y$?
- Qual a relação de cada $X_i$ com $y$?
- Essa relação é linear ou mais complexa?
- Qual o efeito de uma mudança em $X_i$ sobre $y$?

**Exemplo 1: Inferência em regressão linear**  
Um pesquisador deseja entender como fatores socioeconômicos, como renda, escolaridade e idade, influenciam o nível aprovação de políticos (e.g. Governador do estado, Presidente, etc.). Utilizando um modelo de regressão linear, ele pode estimar o efeito de cada variável explicativa sobre a aprovação ($Y$), interpretando os coeficientes para identificar quais fatores têm maior impacto e se essas relações são positivas ou negativas.

**Exemplo 2: Inferência em PLN**  
Em Processamento de Linguagem Natural, um estudo pode investigar quais características textuais estão associadas à viralização de postagens em redes sociais. O pesquisador pode analisar variáveis como o uso de emojis, hashtags, comprimento do texto e presença de palavras-chave, buscando entender como cada uma dessas variáveis ($X_i$) contribui para o número de compartilhamentos ou curtidas ($Y$). O objetivo não é apenas prever se uma postagem será viral, mas explicar quais elementos do texto aumentam ou diminuem essa probabilidade.

**Exemplo 3: Inferência em classificação**  
Outro exemplo é um estudo sobre fatores que influenciam a classificação de notícias como verdadeiras ou falsas. Ao invés de apenas construir um modelo para detectar fake news, o pesquisador pode examinar quais padrões linguísticos, fontes ou estruturas textuais estão estatisticamente associados à veracidade das notícias, permitindo uma compreensão mais profunda dos mecanismos de desinformação.


## Como estimar $f$?

Estabelecemos que, em inferência ou predição, o objetivo principal é estimar a função $\hat{f}$ tal que $y \approx \hat{f}(x)$ para cada par de observações ($x_i, y_i$). Os métodos/modelos que podem fazer essa estimação estão divididos em dois grupos: **Paramétricos** e **Não Paramétricos**.


### Métodos Paramétricos

Métodos paramétricos geralmente operam em dois passos:

1. Definir ou especificar a forma funcional de $f$. Para isso, precisa escolher as variáveis explicativas de interesse, se a relação com $Y$ será linear ou não, se existem interações entre as variáveis independentes, etc. Vamos olhar para uma regressão linear univariada:

$$
Y = \beta_0 + \beta_1*X_1
$$

Talvez você já tenha visto algo muito similar quando estudou a equação reduzida da reta:

$$
Y = mX + b
$$

$b$, ou o coeficiente linear, é o ponto onde a reta intercepta o eixo y. Na regressão linear univariada, estimamos o $\beta_0$, que também é o coeficiente linear. $m$ é o coeficiente angular, que na regressão univariada é representado por $\beta_1$, e mostra o quanto $y$ varia com o aumento de $x$. Essa regressão pode ser generalizada para mais variáveis (ainda sendo uma regressão linear):

$$
Y = \beta_0 + \beta_1*X_1 + \beta_2*X_2 + ... + \beta_p * X_p
$$


Onde $p$ é o número de variáveis a serem incluídas. No exemplo de James et al. ({cite}`james2023introduction`.), temos um modelo sobre a renda em função de anos de estudo e *seniority* (quantos anos o indivíduo trabalha na empresa). 

2. Com as variáveis e forma de $f$ definidas, precisamos escolher um modo de fazer o *fit* do modelo às observações. Isto é, precisamos estimar os parâmetros $\beta_0, \beta_1, ..., \beta_p$. O método mais comum em regressão linear para estimar esses parâmetros é o *OLS*, *Ordinary Least Squares*



```{figure} ../aula5/images/fig2.4.png
---
width: 100%
name: income
align: center
---
Modelo Linear da Relação entre Renda do indivíduo, anos de educação e *seniority*. Fonte: James et al. ({cite}`james2023introduction`., p. 21)
```


A {numref}`Figura {number} <income>` mostra como ficaria um modelo OLS na representação 3D da relação entre Renda, Anos de educação, e senioridade. Apesar de parecer um pouco estranho por estar em três dimensões, essa relação é linear. Analisando o quadrante que mostra a evolução entre renda e anos de educação, parece que á uma relação linear positiva: quanto mais anos de educação, maior a renda. O mesmo parece acontecer com *seniority*. Esse método é paramétrico justamente por que o pesquisador define a forma funcional e como as variáveis $X_i$ se relacionam com a variável explicativa $Y$. Após definir a forma, o pesquisador deve escolher um método para estimar os **Parâmetros** $\beta_0, \beta_1, ..., \beta_p$. 

Entre os métodos paramétricos mais conhecidos estão a regressão linear, a regressão logística, o modelo de Poisson e o modelo de sobrevivência de Cox. Esses métodos assumem uma forma funcional específica para a relação entre as variáveis explicativas e o resultado, permitindo a interpretação direta dos parâmetros estimados e facilitando a análise dos efeitos individuais.


### Métodos Não-Paramétricos

Em contraposição aos métodos paramétricos, os métodos **não-paramétricos** não assumem uma forma funcional de $f$, procurando estimá-lo de forma a chegar bem perto das observações individuais, sem ser muito rígido nem flexível demais. No exemplo da renda do indivíduo, ainda usamos as variáveis de anos de estudo e *seniority*, mas não definimos se essa relação é linear, se há interação entre as variáveis explicativas, etc. A {numref}`Figura {number} <incomenonpar>` mostra como ficaria um modelo não paramétrico para a relação entre Renda do Indivíduo, anos de educação e *seniority* na empresa.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Essas abordagens podem apresentar uma grande vantagem em relação às paramétricas: ao evitar a suposição de uma forma funcional específica para f, têm o potencial de ajustar com precisão uma gama bem mais ampla de formatos possíveis para f. Toda abordagem paramétrica traz consigo a possibilidade de que a forma funcional usada para estimar f seja muito diferente da verdadeira f, caso em que o modelo resultante não descreverá bem os dados. Em contraste, as abordagens não paramétricas eliminam completamente esse risco, já que praticamente não se assume nada sobre a forma de f. Entretanto, as abordagens não paramétricas sofrem de uma grande desvantagem: como não reduzem o problema de estimar f a um pequeno conjunto de parâmetros, é necessário um número muito maior de observações (bem acima do normalmente exigido por uma abordagem paramétrica) para se obter uma estimativa precisa de f."
({cite}`james2023introduction`., p. 22, tradução nossa)
```



```{figure} ../aula5/images/fig2.6.png
---
width: 100%
name: incomenonpar
align: center
---
Modelo Não Paramétrico da Relação entre Renda do indivíduo, anos de educação e *seniority*. Fonte: James et al. ({cite}`james2023introduction`., p. 23)
```


## Trade-off entre Flexibilidade e Interpretabilidade

Lendo a seção anterior, talvez surja a seguinte pergunta: Por que usar métodos paramétricos? O modelo não paramétrico, apesar de precisar de mais dados, parece se ajustar melhor às observações e gera melhores resultados. Vamos ver que, pelo menos na área de aprendizado de máquina, a preferência é por métodos não-paramétricos. No entanto, métodos paramétricos são mais interpretáveis e adequados para a inferência: Com eles é possível entender quanto o aumento de um ano de estudo pode impactar a renda de um indivíduo no futuro, por exemplo. A {numref}`Figura {number} <flexinter>` mostra como varia a interpretabilidade ao longo da flexibilidade. Ao longo do curso, veremos modelos OLS (ou lineares, como a regressão logística), modelos baseados em árvores (Decision Trees, Random Forests), *bagging* e *boosting*, *SVM* (*Support Vector Machines*) e alguns modelos de *Deep Learning*.

```{figure} ../aula5/images/fig2.7.png
---
width: 100%
name: flexinter
align: center
---
Representação do Trade-off entre interpretabilidade e flexibilidade. Fonte: Id., (p. 24)
```


Veremos que quanto mais flexível um modelo, mais difícil de entender como cada *feature* impacta na qualidade das previsões, chegando ao ápice em modelos de *deep learning*, que veremos ao final do curso. Métodos paramétricos, mais flexíveis, geralmente estimam um $f$ que se aproxima mais dos valores reais de $y$, mas isso vem com o custo para a interpretabilidade. Por isso, é importante tentar utilizar métodos paramétricos mais simples se o intuito é o de entender como cada variável impacta o modelo, seja no contexto de predição ou no contexto de inferência.

Apesar disso, existem movimentos que buscam conciliar interpretabilidade e flexibilidade, como o do *Interpretable Machine Learning*. Avanços estão sendo feitos para permitir que seja possível investigar a "caixa preta" de modelos mais flexíveis, como o deep learning, e entender o impacto de cada *feature* ou conjunto de palavras na previsão dos modelos.

Além disso, nem sempre o modelo mais flexível será mais generalizável para outras amostras para além do treinamento. A {numref}`Figura {number} <flexteste>` mostra como diferentes modelos performam no treinamento e teste em dados simulados (ou seja, em que se sabe o verdadeiro *DGP*).



```{figure} ../aula5/images/fig2.9.png
---
width: 100%
name: flexteste
align: center
---
À esquerda: dados simulados a partir de f, mostrados em preto. Três estimativas de f são exibidas: a linha de regressão linear (curva laranja) e dois ajustes por splines de suavização (curvas azul e verde).

À direita: Erro Quadrático Médio de treinamento (curva cinza), EQM de teste (curva vermelha) e EQM mínimo possível de teste entre todos os métodos (linha tracejada). Os quadrados representam os EQMs de treinamento e de teste para os três ajustes mostrados no painel da esquerda. Fonte: Id., p. 29.
```

O verdadeiro $f(x)$ da população está representado na linha preta, e seria o modelo que queremos alcançar. No entanto, na prática não sabemos o verdadeiro $f(x)$, então só podemos comparar a performance dos modelos em algumas situações: **Treinamento, Teste e Validação**. Na {numref}`Figura {number} <flexteste>` temos a comparação dos modelos no bancos de treinamento e teste (veremos o que é a validação nas próximas aulas). O modelo de regressão linear, menos flexível, traça uma reta ao longo das observações por meio do cálculo do *OLS*, gerando bastante erro entre os valores preditos e os valores reais. Por isso, apresenta os maiores erros de treinamento e teste (Quadrado laranja na figura à direita). O modelo de regressão linear apresenta *underfitting*, tendo resultados ruins no treinamento e no teste.


 Na curva verde, temos um modelos flexível que se ajusta bem de perto às observações, gerando pouquíssimo erros no treinamento (quadrado verde). No entanto, apresenta alto erro no banco de teste. Isso se deve ao fato de que se ajustou muito bem aos dados de treinamento, mas não é generalizável para outras amostras. Isso é chamado de *overfitting*.

 Por fim, temos a linha azul. Ela é bem próxima da verdadeira função $f$ representada na linha preta. O modelo se ajusta bem aos dados de treinamento, apresentando baixo erro nesse conjunto. Dos três modelos, é o que apresenta também o menor erro de teste. A linha azul representa a situação ideal em um modelo de aprendizado de máquina: tem bons resultados no treinamento e no teste.


 ```{admonition} 💬 Com a palavra, os autores:
:class: quote
"À medida que a flexibilidade do modelo aumenta, o EQM de treinamento diminui, mas o EQM de teste pode não acompanhar essa queda. Quando um método produz um EQM de treinamento pequeno, mas um EQM de teste grande, dizemos que ele está sofrendo overfitting (ajuste excessivo) aos dados. Isso ocorre porque o procedimento de aprendizado estatístico se empenha demais em encontrar padrões no conjunto de treinamento e acaba capturando alguns que surgem apenas por acaso, e não por características reais da função desconhecida f. Ao fazermos overfitting nos dados de treinamento, o EQM de teste fica muito alto, pois os supostos padrões detectados no treinamento simplesmente não existem no conjunto de teste."
({cite}`james2023introduction`., pp. 30-31, tradução nossa)
```


## Tradeoff entre Viés e Variância

Essa relação entre flexibilidade do modelo e os erros de treinamento e teste se devem à duas propriedades de métodos de aprendizado estatístico: **Viés e Variância**. Para minimizar o erro esperado do teste, precisamos deu ma técnica de aprendizado de máquina que simultaneamente tenha baixa variância e baixo viés. O seguinte [site](https://mlu-explain.github.io/bias-variance) apresenta uma interessante visualização deste problema. Aqui estão algumas definições gerais:

- **Variância**: Grosso modo, podemos definir a variância como o quanto a função estimada $\hat{f}$ mudaria se alterássemos os dados de treinamento. Idealmente, deveríamos ter baixa variância. Isto é, ao alterar os dados de treinamento, os resultados não variariam muito. Métodos mais flexíveis geralmente apresentam maior variância, e por isso podem apresentar melhores resultados no banco de treinamento do que no de teste. Portanto, são mais propensos ao *overfitting*.
- **Viés**: Viés é o erro introduzdio pela aproximação de um problema da vida real por um modelo simples. Por exemplo, tentar modelar a renda de um indíviduo como uma relação linear com anos de estudo e *seniority*. Métodos mais flexíveis geralmente apresentam menor viés. Métodos menos flexíveis (como a regressão linear) são mais propensos ao viés e, consequentemente, ao *underfitting*.  



 ```{admonition} 💬 Com a palavra, os autores:
:class: quote
"De modo geral, métodos mais flexíveis produzem menor viés. Como regra geral, à medida que recorremos a métodos mais flexíveis, a variância aumenta e o viés diminui. A taxa relativa de variação dessas duas quantidades determina se o MSE de teste aumenta ou diminui. Quando ampliamos a flexibilidade de uma classe de métodos, o viés tende a diminuir inicialmente mais rápido do que a variância aumenta, fazendo com que o MSE de teste esperado caia. Contudo, em certo ponto, incrementar ainda mais a flexibilidade tem pouco efeito sobre o viés, mas passa a aumentar significativamente a variância; nessa fase, o MSE de teste volta a crescer."
({cite}`james2023introduction`., pp. 30-31, tradução nossa)
```

A {numref}`Figura {number} <biasvartrade>` mostra uma representação teórica de como opera a relação entre complexidade do modelo e erro de predição nos bancos de treinamento e teste. No geral, a ideia é encontrar um modelo que consegue bons resultados no treinamento, mas que generalize bem para outros bancos (Teste e Validação). Essa figura foi retirada de outro livro feita pelos mesmos autores do ISLP: The Elements of Statistical Learning, de Hastie, Tibshirani e Friedman ({cite}`hastie2009elements`.).


```{figure} ../aula5/images/biasvarESL.png
---
width: 100%
name: biasvartrade
align: center
---
Erro de treino e de teste como uma função da complexidade do modelo. Fonte: Hastie, Tibshirani e Friedman ({cite}`hastie2009elements`., p. 38)
```

Podemos representar o erro de um modelo da seguinte forma:

$$
Erro = Viés² + Variância + \epsilon
$$

Como vimos anteriormente, não há muito o que fazer sobre $\epsilon$, pois ele representa o erro aleatório e irredutível:


$$
+ \underbrace{Var(\epsilon)}_{\text{Erro irredutível: variabilidade aleatória dos dados}}
$$


Então o foco fica em encontrar um equilíbrio no resto da equação, tentando reduzir tanto viés quanto variância. Como podemos fazer isso?


## "Protocolo Padrão" de Aprendizado de Máquina

Para garantir que o pesquisador possui o melhor modelos e resultados, além de apresentar a variância e viés claramente para os leitores, propõe-se o seguinte "Protocolo Padrão":



```{admonition} 📋 Protocolo de Aprendizado de Máquina
:class: exercise

Podemos propor o seguir protocolo:

1. **Definição do Problema**  
   - Identifique um problema de pesquisa relevante na sua área de interesse. Por exemplo, "Como as redes sociais influenciam o debate público sobre mudanças climáticas?".
   - Qual será o objetivo da pesquisa?
   - Quem é a população de interesse?
   - Que universo de documentos irei estudar?

   
2. **Coleta de Dados** 

- a) Criação de um banco próprio:
   - Que tipo de dados textuais você utilizaria para abordar esse problema? Considere fontes como redes sociais, discursos políticos, artigos de jornal, etc.
   -

3. **Método**  
   - Qual método de análise você aplicaria para identificar padrões ou temas nos textos? Exemplos incluem Latent Dirichlet Allocation (LDA), análise de sentimentos ou classificação supervisionada.

4. **Validação**  
   - Como você validaria os resultados da sua análise? Pense em estratégias como leitura manual de amostras, comparação com eventos conhecidos ou validação cruzada.

5. **Inferência**  
   - Que tipo de inferência você poderia fazer com base nos resultados? Por exemplo, estimar o impacto de uma política pública ou identificar mudanças no discurso político ao longo do tempo.

6. **Reflexão Final**  
   - Como o ciclo iterativo de descoberta e mensuração pode ajudar a refinar suas perguntas de pesquisa e hipóteses iniciais? Considere como os dados podem influenciar o foco do seu estudo.

Após responder às perguntas, discuta suas respostas com um colega ou no grupo de estudos. Reflita sobre como o paradigma proposto por Grimmer et al. pode ser aplicado para enriquecer sua pesquisa.
```

