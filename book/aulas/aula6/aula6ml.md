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

Os parâmetros $\beta_0$ e $\beta_1$ também são estimados, assim como na regressão linear. A diferença está em como é feito. Na regressão linear, utilizamos o método de mínimos quadrados ordinários (ou *OLS* em inglês) para estimar os parâmetros da equação. Aqui, utilizaremos o método da Máxima Verossimilhança, ou *Maximum Likelihood*, que veremos na próxima subseção (e coloquei um vídeo complementar para quem tiver interesse). Com um pouco de manipulação (segundo os autores, não eu), chegamos em:

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

A máxima verossimilhança é uma função utilizada em muitos modelos paramétricos não-lineares, e com os coeficientes estimados por ela podemos fazer previsões para dados não vistos.

### Regressão Logística Múltipla

A Regressão Logística Múltipla é a generalização da regressão logística com outcome binário (sim ou não) para mais variáveis preditoras. Nesse cenário, o *log odds* passa a ser calculado por:

$$
\log\!\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X_1 + ... + \beta_p* X_p
$$

Onde X = ($X_1, ..., X_p$) são os preditores. Da mesma forma que antes, o método de verossimilhança é utilizado para estimar os parâmetros $\beta_0, \beta_1, ...  ,\beta_p$.


### Regressão Logística Multinomial

Até agora, trabalhamos com o caso de um outcome *Y* binário (sim ou não, 0 ou 1). No entanto, em muitos casos estamos interessados em classificar mais de uma categoria/classe. Para tarefas em que o número K de classes é $>2$, utilizamos o *Multinomial Logit*, ou **Regressão Logística Multinomial**, que é uma extensão da regressão logística para mais classes. Nessa extensão, uma das classes será utilizada como base de comparação para estimar os parâmetros. $p(X)$ é alterado da seguinte maneira:


$$
Pr(Y_i = K | X = x)
$$

Ou seja, a probabilidade de que a observação individual $Y_i$ seja de determinada categoria K, dado os valores das variáveis preditoras. Para estimar esse novo $p(X)$, estimamos

$$
\Pr\bigl(Y = k \mid X = x\bigr)
  = \frac{
        e^{\beta_{k0} + \beta_{k1}x_1 + \cdots + \beta_{kp}x_p}
      }{
        1 \;+\; \displaystyle\sum_{l=1}^{K-1}
              e^{\beta_{l0} + \beta_{l1}x_1 + \cdots + \beta_{lp}x_p}
      }.
$$

Que pode ser lida assim:

A **probabilidade** de um indivíduo pertencer à categoria *k* (entre K possíveis) **dado** o vetor de preditores $x=(x_1,\dots,x_p)$ é igual à razão entre   o **peso exponencial** atribuído a essa categoria obtido somando o intercepto $\beta_{k0}$ ao efeito de cada preditor $x_j$ ponderado pelo seu coeficiente $\beta_{kj}$ e a soma desse mesmo peso **mais** os pesos de todas as demais categorias tomadas como comparação.

Em outras palavras:

1. Para cada classe *k* calculamos um escore linear
$\beta_{k0} + \beta_{k1}x_1 + \dots + \beta_{kp}x_p$.
2. Transformamos esse escore em algo estritamente positivo aplicando a exponencial $e^{(\cdot)}$; isso garante que valores maiores de escore se convertam em pesos maiores.
3. A probabilidade final de estar na classe *k* é esse peso dividido pela soma de:
    - 1 (peso da classe-de-referência implicitamente tratada como $ \beta_{00}=0 $) **mais**
    - os pesos de todas as outras K−1 classes explicitadas no denominador.

Assim, o modelo:

- Mantém todas as probabilidades no intervalo 0–1.
- Faz com que a soma das probabilidades sobre todas as K classes seja 1.
- Permite interpretar cada $\beta_{kj}$ como o efeito de $x_j$ na chance logarítmica de estar na classe *k* em comparação com a classe-referência.

O *log odds* passa a ser


$$
\log\!\left(
      \frac{\Pr\!\bigl(Y = k \mid X = x\bigr)}
           {\Pr\!\bigl(Y = K \mid X = x\bigr)}
    \right)
  \;=\;
  \beta_{k0} + \beta_{k1}x_1 + \cdots + \beta_{kp}x_p.
$$

Onde o logaritmo da probabilidade de pertencer à classe $k$ em comparação com as outras classes é igual à uma equação linear com os preditores. A decisão da classe a ser utilizada como base de comparação é



```{admonition} 💬 Com a palavra, os autores:
:class: quote
"irrelevante. Por exemplo, ao classificar atendimentos de emergência em AVC, overdose de drogas e crise epiléptica, suponha que ajustemos dois modelos de regressão logística multinomial: um tomando AVC como referência e outro tomando overdose de drogas como referência. As estimativas dos coeficientes diferirão entre os dois modelos ajustados devido à escolha distinta de referência, mas os valores ajustados (previsões), os log-odds entre qualquer par de classes e os demais resultados importantes do modelo permanecerão iguais. Ainda assim, a interpretação dos coeficientes em um modelo de regressão logística multinomial deve ser feita com cuidado, pois ela depende da categoria de referência."
({cite}`james2023introduction`., p. 145, tradução nossa)
```


## Modelos Generativos para Classificação

Os autores apresentam uma segunda classe de modelos comuns utilizados na classificação: os modelos generativos. Modelos generativos são chamados assim porque buscam modelar explicitamente o processo de geração dos dados. Em vez de apenas aprender a relação direta entre as variáveis explicativas ($X$) e a variável resposta ($Y$), como fazem os modelos discriminativos, os modelos generativos aprendem a distribuição conjunta $P(X, Y)$ ou, de forma equivalente, $P(X|Y)$ e $P(Y)$. Isso permite que eles não só classifiquem exemplos, mas também simulem ou gerem novos dados que seguem o mesmo padrão observado. Na prática, modelos generativos como Naive Bayes e LDA (Linear Discriminant Analysis) estimam como as características dos dados são distribuídas dentro de cada classe e, a partir disso, calculam a probabilidade de um exemplo pertencer a cada categoria. Essa abordagem é útil para entender melhor a estrutura dos dados e pode ser empregada em tarefas como classificação, detecção de anomalias e geração de exemplos sintéticos.


Nesse tipo de modelos, podemos modelar a distribuição dos $p$ preditores $X$ separadamente para cada classe em $Y$. com isso, usamos o **Teorema de Bayes** para obtermos as estimativas de $Pr (y=k | X= x)$.

### Por que não Regressão Logística?

- Quando há separação substantiva entre as classes, as estimativas do *logit* podem ser instáveis;
- Se a distribuição dos preditores $X$ for **aproximadamente** normal, os métodos generativos serão mais precisos;
- Esses métodos se extendem naturalmente para um número de classes $K >= 2$

### Teorema de Bayes

Suponha que queremos classificar uma observação entre uma em K classes, onde $K >= 2$. Sendo $\pi k$ a representação da probabilidade *a priori* de que uma observação escolhida aleatoriamente venha da $k_{ésima}$ classe. E sendo $fk(x) = Pr(X|y = k)$ a função de densidade de  X para uma observação da da $k_{ésima}$ classe. Então, o teorema de Bayes estabelece que:

$$
\Pr\bigl(Y = k \mid X = x\bigr)
  = \frac{\pi_k\,f_k(x)}
         {\displaystyle\sum_{l=1}^{K} \pi_l\,f_l(x)}.
$$

A leitura "intuitiva" é a seguinte: "Pegue o quão comum cada classe é na população (o peso πₖ) e multiplique por quão bem as características x se encaixam nessa classe (a verossimilhança fₖ(x)). Depois compare esse peso com a soma dos pesos de todas as classes. A fração resultante é exatamente a probabilidade de que a observação pertença à classe k."

Com isso temos a probabilidade posterior $pk(x)$ = $Pr(y= k | X= x)$, que é a probabilidade de que uma observação pertence à classe k, dado os valores dos preditores para aquela observação. Os modelos dessa parte do capítulo todos vão utilizar o teorema de Bayes como parte das estimativas das probabilidades $pk(x)$.

### *Linear Discriminant Analysis* (LDA)

O *Linear Discriminant Analysis* (LDA) é um modelo generativo utilizado para tarefas de classificação, especialmente quando a variável resposta possui duas ou mais categorias. O LDA parte do princípio de que os dados de cada classe seguem uma distribuição normal multivariada com médias diferentes, mas compartilham a mesma matriz de covariância. Ou seja, ele assume que, dentro de cada classe, as variáveis explicativas ($X$) têm distribuição aproximadamente normal e que a dispersão dos dados é semelhante entre as classes.

O funcionamento do LDA envolve dois passos principais: primeiro, ele estima a média e a variância das variáveis explicativas para cada classe, além das probabilidades a priori de cada classe na população. Em seguida, utiliza o Teorema de Bayes para calcular a probabilidade de uma nova observação pertencer a cada classe, combinando a verossimilhança dos dados com o peso de cada classe.

A fronteira de decisão do LDA entre as classes é linear, pois o modelo constrói uma combinação linear das variáveis explicativas para separar as categorias. Isso significa que o LDA busca encontrar a linha (ou hiperplano, em dimensões maiores) que melhor discrimina entre as classes, maximizando a separação entre elas e minimizando a dispersão dentro de cada classe.

O LDA é especialmente útil quando as suposições de normalidade e covariância igual são razoáveis, e pode ser aplicado em problemas como reconhecimento de padrões, classificação de textos, diagnóstico médico e análise de crédito. Além de classificar novas observações, o LDA também permite interpretar quais variáveis são mais importantes para distinguir entre as classes, fornecendo insights sobre a estrutura dos dados.


### *Quadratic Discriminant Analysis* (QDA)

O *Quadratic Discriminant Analysis* (QDA) é uma extensão do LDA que relaxa uma das principais suposições do modelo: enquanto o LDA assume que todas as classes compartilham a mesma matriz de covariância, o QDA permite que cada classe tenha sua própria matriz de covariância. Isso significa que o QDA pode capturar situações em que a dispersão ou a forma das distribuições das variáveis explicativas ($X$) é diferente entre as classes.

No QDA, os dados de cada classe ainda são modelados como provenientes de uma distribuição normal multivariada, mas agora cada classe pode ter uma dispersão e correlação entre variáveis próprias. Como resultado, a fronteira de decisão entre as classes deixa de ser linear e passa a ser quadrática, permitindo separar classes que têm formatos ou distribuições mais complexas.

O funcionamento do QDA envolve estimar, para cada classe, a média das variáveis explicativas, a matriz de covariância específica e a probabilidade a priori. Utilizando o Teorema de Bayes, o QDA calcula a probabilidade de uma nova observação pertencer a cada classe, levando em conta as diferenças na dispersão dos dados.

O QDA é especialmente útil quando as classes apresentam padrões de variabilidade distintos, como em problemas de classificação de imagens, reconhecimento de padrões ou situações em que a estrutura dos dados é mais heterogênea. Por ser mais flexível que o LDA, o QDA pode se adaptar melhor a dados complexos, mas também exige mais dados para estimar corretamente as matrizes de covariância de cada classe.

### LDA ou QDA?


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Por que importa se assumimos ou não que as K classes compartilham uma matriz de covariância comum? Em outras palavras, por que alguém preferiria LDA a QDA, ou vice-versa? A resposta está no trade-off viés-variância. Quando há p preditores, estimar uma matriz de covariância requer estimar p(p+1)/2 parâmetros. O QDA estima uma matriz de covariância separada para cada classe, somando Kp(p+1)/2 parâmetros. Com 50 preditores, isso corresponde a múltiplos de 1 275, ou seja, muitos parâmetros. Ao assumir que as K classes compartilham uma matriz de covariância comum, o modelo LDA torna-se linear em x, o que implica Kp coeficientes lineares a estimar. Consequentemente, o LDA é um classificador muito menos flexível que o QDA e, portanto, tem variância substancialmente menor. Isso pode levar a um desempenho de previsão melhor. Mas há um trade-off: se a suposição de que as K classes compartilham uma matriz de covariância comum estiver muito errada, o LDA pode sofrer de alto viés. De modo geral, o LDA tende a ser uma aposta melhor que o QDA quando há poucas observações de treino e, portanto, reduzir a variância é crucial. Em contraste, o QDA é recomendado se o conjunto de treino for muito grande, de modo que a variância do classificador não seja uma grande preocupação, ou se a suposição de uma matriz de covariância comum para as K classes for claramente insustentável."
({cite}`james2023introduction`., p. 157, tradução nossa)
```

### *Naive Bayes*


O *Naive Bayes* é outro modelo generativo amplamente utilizado em tarefas de classificação, especialmente em Processamento de Linguagem Natural. Sua principal característica é a suposição de independência condicional entre as variáveis explicativas ($X$) dado a classe ($Y$). Ou seja, o modelo assume que, dentro de cada classe, as variáveis são estatisticamente independentes entre si — uma simplificação que raramente é verdadeira na prática, mas que torna o modelo extremamente eficiente e fácil de implementar.

O funcionamento do Naive Bayes envolve calcular, para cada classe, a probabilidade a priori ($P(Y)$) e a probabilidade de observar cada valor das variáveis explicativas dado a classe ($P(X_i|Y)$). Utilizando o Teorema de Bayes, o modelo combina essas probabilidades para estimar a probabilidade de uma nova observação pertencer a cada classe. Apesar da suposição "ingênua" de independência, o Naive Bayes costuma apresentar bom desempenho em problemas de texto, como classificação de e-mails em spam ou não spam, análise de sentimentos e categorização de documentos.

Além de ser rápido e escalável para grandes volumes de dados, o Naive Bayes é robusto a dados faltantes e pode ser facilmente adaptado para diferentes tipos de variáveis (binárias, categóricas ou contínuas). Em resumo, o Naive Bayes oferece uma solução prática e eficiente para problemas de classificação, especialmente quando a simplicidade e a velocidade são prioridades.


## Conclusão

Neste capítulo, aprofundamos o entendimento sobre o problema de classificação em aprendizado supervisionado, destacando as limitações da regressão linear para variáveis categóricas e a importância de utilizar métodos apropriados para tarefas de classificação. Exploramos a regressão logística, suas extensões para múltiplos preditores e múltiplas classes, e discutimos o papel do logit como ligação entre variáveis explicativas e probabilidades. Apresentamos também os modelos generativos, como LDA, QDA e Naive Bayes, que modelam explicitamente o processo de geração dos dados e utilizam o Teorema de Bayes para estimar probabilidades de pertencimento às classes. Discutimos os pressupostos, vantagens e limitações de cada abordagem, bem como o trade-off entre viés e variância na escolha do modelo. Por fim, reforçamos a importância de compreender as características dos dados e dos métodos para realizar classificações precisas, interpretáveis e adequadas ao contexto de cada problema.

## Notas

[^1]: **Classificadores** são modelos de aprendizado de máquina supervisionado projetados para atribuir exemplos a categorias ou classes distintas com base em suas características. Eles são utilizados quando a variável resposta é categórica, como na identificação de sentimentos em textos, classificação de imagens ou detecção de spam em e-mails.