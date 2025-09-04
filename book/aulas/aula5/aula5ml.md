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

Em contraposição aos métodos paramétricos, os métodos **não-paramétricos** não assumem uma forma funcional de $f$, procurando estimá-lo de forma a chegar bem perto das observações individuais, sem ser muito rígido nem flexível demais. No exemplo da renda do indivíduo, ainda usamos as variáveis de anos de estudo e *seniority*, mas não definimos se essa relação é linear, se há interação entre as variáveis explicativas, etc.


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
















