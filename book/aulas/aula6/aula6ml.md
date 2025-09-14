# O Problema de Classificação

Na última aula, aprofundamos os conceitos fundamentais do aprendizado supervisionado, diferenciando os objetivos de inferência e predição, e discutindo como construir bancos de dados de treinamento confiáveis para aplicações em PLN. Exploramos o papel dos métodos paramétricos e não-paramétricos, os principais trade-offs entre flexibilidade e interpretabilidade, e a importância de equilibrar viés e variância para obter modelos robustos e generalizáveis. Também revisamos métricas essenciais para avaliação de classificadores e destacamos a necessidade de testar os modelos em dados novos para garantir sua utilidade prática. Por fim, apresentamos um protocolo padrão para conduzir pesquisas rigorosas e transparentes em aprendizado de máquina supervisionado.

Na aula de hoje, iremos discutir o problema específico de classificação, e alguns dos modelos mais básicos utilizados para esta tarefa. O problema de classificação surge quando temos uma variável categórica como nossa variável resposta $y$. Ou seja, não queremos prever um valor numérico contínuo (e.g. valor de uma casa), mas uma classe (favorável, desfavorável, incerto). Alguns dos classificadores[^1] mais comuns são: Regressão Logística, *Linear Discriminant Analysis* (*LDA*), *Quadratic Discriminant Analysis*, *Naive Bayes* e *K Nearest Neighbors* (KNN). No capítulo 2, James et al. ({cite}`james2023introduction`.) discutem o KNN, e no capítulo 4 focam nos outros citados acima.


## Por que não Regressão Linear?


Uma questão que pode surgir é a de por que não usar a regressão linear para classificação se podemos colocar as categorias como números? Vamos supor o seguinte caso de classificação em três diagnósticos:

$$
Y =
\begin{cases}
  1 & \text{se AVC;} \\\\
  2 & \text{se Overdose;} \\\\
  3 & \text{se Crise Epiléptica.}
\end{cases}
$$

Além de gerar um ordenamento entre os casos (Crise Epiléptica ser "maior" que AVC), estabelece que a distância entre um AVC e a overdose é a mesma que entre uma overdose e uma crise epiléptica. Ainda por cima, alterar a ordem dessa categorização geraria estimativas com significados e dimensões muito diferentes, tornando o modelo de regressão linear instável e pouco confiável. A situação melhora um pouco quando temos apenas dois resultados possíveis:


$$
Y =
\begin{cases}
  0 & \text{se AVC;} \\\\
  1 & \text{se Overdose;} \\\\
\end{cases}
$$

Mesmo se alterássemos os valores, os resultados se manteriam. No entanto, poderíamos obter valores estimados para além dos limites 0 e 1, além de obter poucas estimativas para casos mais perto dos valores máximos e mínimos, como mostra a seguinte figura:


```{figure} ../aula6/images/fig4.2.a.png
---
width: 100%
name: reglinclass
align: center
---
Classificação no banco "Default" utilizando uma regressão linear. Fonte: James et al. ({cite}`james2023introduction`., p. 139)
```

A {numref}`Figura {number} <reglinclass>` mostra que a regressão linear (linha azul) concentra a maior parte dos valores estimados de Y (Probabilidade de *Default*) bem perto de zero. Portanto, pouquíssimos indivíduos seriam classificados como inadimplentes (ou devedores). Além disso, encontramos probabilidades negativas (o que é impossível) perto de valores de *balance* (Saldo do cartão de crédito) menores que 500. No laboratório de hoje exploraremos um pouco mais desse banco de dados apresentado pelos autores, e tentaremos classificar os adimplentes e inadimplentes utilizando os diversos modelos discutidos no capítulo.

```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Para resumir, existem pelo menos duas razões para não realizar classificação utilizando um método de regressão [linear]: (a) um método de regressão não pode acomodar uma resposta qualitativa com mais de duas classes; (b) um método de regressão não fornecerá estimativas significativas de Pr(Y | X), mesmo com apenas duas classes. Assim, é preferível usar um método de classificação que seja realmente adequado para valores de resposta qualitativa."
({cite}`james2023introduction`., p. 138, tradução nossa)
```

## A Regressão Logística

Quando temos um resultado binário (Sim ou não, 0 ou 1), podemos utilizar a regressão logística para modelar a probabilidade de que $Y_i$ pertence a determinada categoria.

$$
Pr(Y_i = 1 | X)
$$

Traduzindo, queremos a probabilidade ($Pr$) de que $Y_i$ pertença a categoria 1 dado ($|$) os valores das variáveis preditoras associadas àquela observação ($X$). No caso do banco de inadimplentes (*Default*), podemos querer saber a probabilidade de que um indivíduo vai ser inadimplente dada suas características preditoras (Se é estudante ou não, renda, dívidas anteriores, etc.). No caso da regressão logística simples (de um único preditor), podemos pensar somente com relação ao saldo (*Balance*) do cartão do indivíduo:


$$
Pr(Inadimplente = Sim | Saldo)
$$

Estimando a mesma relação apresentada na {numref}`Figura {number} <reglinclass>` com um modelo de regressão logística, obtemos o seguinte resultado:

```{figure} ../aula6/images/fig4.2.b.png
---
width: 100%
name: reglogclass
align: center
---
Classificação no banco "Default" utilizando uma regressão logística. Fonte: James et al. ({cite}`james2023introduction`., p. 139)
```

A {numref}`Figura {number} <reglogclass>` mostra que temos uma relação muito mais "limpa" entre o saldo de cartão de crédito e os valores estimados para a probabilidade de que seja um inadimplente: Não possuímos valores negativos na função estimada (curva azul), e indivíduos com maior saldo de cartão tem maior probabilidade de serem classificados como inadimplentes. Para modelar a $Pr(Y_i = 1 | X)$ na regressão logística, ou $p(X)$ para encurtar, precisamos da **Função Logística**, uma das funções que permitem um *output* entre zero e um.


### Função Logística

Lembrando que $p(x)$ é equivalente à $Pr(Y_i = 1 | X)$, podemos estimar a regressão logística utilizando a seguinte **Função Logística**

$$
p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}.
$$

Os parâmetros $\beta_0$ e $\beta_1$ também são estimados, assim como na regressão linear. A diferença está em como é feito. Na regressão linear, utilizamos o método de mínimos quadrados ordinários (ou *OLS* em inglês) para estimar os parâmetros da equação. Aqui, utilizaremos o método da Máxima Verossimilihança, ou *Maximum Likelihood*, que veremos na próxima subseção (e coloquei um vídeo complementar para quem tiver interesse). Com um pouco de manipulação (segundo os autores, não eu), chegamos em:

$$
\frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1 X}.
$$

O lado esquerdo da equação ($\frac{p(X)}{1 - p(X)}$) é conhecido por *odds*, e podem ter qualquer valor entre 0 e $\infty$, e quanto maior, maior a probabilidade de Inadimplência (no nosso exemplo anterior), e vice-versa. Tirando o logaritmo de ambos os lados, chegamos em.


$$
\log\!\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X.
$$

Que é o *log odds* ou *logit*, este último que é muitas vezes usado como sinônimo de regressão logística. Em um modelo de regressão logística, aumentar $X_1$ em uma unidade altera o valor de *log odds* por $\beta_1$. No entanto a relação entre $p(X)$ e $X$ não é linear, e o quanto $p(x)$ muda com a mudança de $X$ depende do valor atual de $X$.


Em resumo, a regressão logística transforma a probabilidade de um evento em uma escala que pode ser modelada linearmente, utilizando o logit (ou log odds) como ligação entre as variáveis explicativas e o resultado. O modelo estima a relação entre os preditores e a chance de ocorrência de um evento, garantindo que as previsões estejam sempre entre 0 e 1. Essa abordagem é especialmente útil para problemas de classificação binária, pois permite interpretar diretamente o impacto de cada variável sobre a probabilidade do evento e evita problemas comuns da regressão linear, como previsões fora do intervalo válido de probabilidades.



### Método de Máxima Verossimilhança

Para estimar os parâmetros $\beta_0$ e $\beta_1$ na equação
 
$$
p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}.
$$

é utilizado o método de máxima verossimilhança. A intuição por trás desse método é a de ele procura estimar os parâmetros $\beta_0$, $\beta_1$,..., $\beta_p$ (para o caso com mais variáveis preditoras) tal que a probabilidade $\hat{p}(X_i)$ para cada indivíduo corresponda, da melhor maneira possível, à probabilidade observada $p(x_i)$. Ou seja,


$$
\frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} = \hat{p}(X_i) \approx p(X_i) 
$$

Para isso, é utilizada a seguinte função de verossimilhança.

$$
\ell(\beta_0, \beta_1)
     = \prod_{i:\,y_i = 1} p(x_i)\;
       \prod_{i':\,y_{i'} = 0} \!\bigl(1 - p(x_{i'})\bigr).
$$

Em palavras simples, a equação afirma: “Para um conjunto de parâmetros ($\beta_0, \beta_1$), a verossimilhança é o produto da probabilidade prevista do evento em todos os casos que de fato ocorreram ($y = 1$) multiplicado pela probabilidade prevista do não-evento em todos os casos que não ocorreram ($y = 0$).”


Quando as observações são independentes, a verossimilhança de um modelo é obtida multiplicando as probabilidades individuais atribuídas a cada dado observado. Aqui, p(xᵢ) representa a probabilidade calculada pelo modelo (por exemplo, a saída da regressão logística) de que o i-ésimo indivíduo tenha y = 1. Para cada yᵢ = 1, incluímos p(xᵢ) no produto; para cada yᵢ = 0, incluímos 1 − p(xᵢ). Dessa forma, parâmetros que atribuem alta probabilidade aos resultados realmente vistos tornam o produto – e, portanto, a verossimilhança – maior.


[^1]: **Classificadores** são modelos de aprendizado de máquina supervisionado projetados para atribuir exemplos a categorias ou classes distintas com base em suas características. Eles são utilizados quando a variável resposta é categórica, como na identificação de sentimentos em textos, classificação de imagens ou detecção de spam em e-mails.