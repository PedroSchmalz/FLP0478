# Classificação com Texto

Discutimos ao longo do curso que a tarefa de classificação envolve pegar preditores $X$ (ou *features*) e utilizá-los para tentar prever à que categoria $Y$, nosso *target* pertence. Para a classificação com texto, a principal mudança é a de que não utilizaremos mais dados numéricos (Saldo do cartão, variáveis de saúde), e sim **Texto**, pré-processado e representado numericamente. Um exemplo clássico dentro da classificação com texto é a de verificar se uma avaliação de um produto é positiva ou negativa. Isso é uma tarefa dentro da área de Análise de Sentimentos, e as categorias podem variar um pouco. Nessa tarefa, pegamos o texto da avaliação {"O produto é muito bom!", "Não gostei nem um pouco"} e tentaremos classificar eles como positivos ou negativos. Como vimos antes, um classificador básico é a regressão logística, e ela servirá de base para entendermos o mecanismo por trás dos modelos de aprendizado supervisionado com texto.


## Classificação com Regressão Logística

No ISL ({cite}`james2023introduction`.), os autores explicam a Regressão Logística por uma linguamge mais estatística. Já Jurafsky e Martin ({cite}`jurafsky2024speech`.) vão por uma linha mais do Aprendizado de Máquina. Por isso, não estranhem a mudança de jargão e de termos para se referir à Regressão Logística. 

### Função Sigmóide

Como dito antes, o objetivo da regressão logística binária é treinar um classificador capaz de tomar uma decisão binária sobre a classe de uma nova observação de entrada. Aqui introduzimos o classificador sigmoide, que nos ajudará a tomar essa decisão. Em Jurafsky e Martin, a Regressão logística resolve o problema de estimar $Pr(Y=1|X)$ (A probabilidade de que a observação pertence à classe 1, dado os preditores) estimando um vetor de **pesos** e **termos de viés**. Vamos chamar os pesos de $w$ e o viés de $b$. Cada peso $w_i$ é um número real e está associado à um dos preditores $X_i$. O peso representa quão importante é cada preditor para a decisão de classificação, podendo ser negativo ou positivo. Em uma tarefa de classificação de sentimento, provávelmente a palavra "ótimo" terá um peso positivo, e a palavra "horrível" terá um peso negativo. O termo de viés, ou intercepto, é um número real que é adicionado aos *inputs* ao final do cálculo. Para fazer uma decisão em uma observação de teste (após o treinamento), o classificador logístico vai multiplicar cada preditor $X_i$ pelo seu peso $w_i$, e somar isso com o termo de viés. Formalmente, temos:

$$
z \;=\; \left(\sum_{i=1}^{n} w_i x_i\right) + b
$$

No entanto, os valores de $z$ resultantes dessa equação não estão obrigatoriamente entre 0 e 1. Para que possamos calcular a probabilidade de que uma observação é da classe 0 ou 1, precisamos limitar z para esse intervalo. Para isso, passaremos z pela função sigmóide $\sigma$, ou função logística. Com isso, conseguiremos obter probabilidades para cada classe entre 0 e 1.

$$
\sigma(z)
   \;=\;
   \frac{1}{1 + e^{-z}}
   \;=\;
   \frac{1}{1 + \exp(-z)}
$$

 A {numref}`Figura {number} <sigmoide>` ilustra como a função sigmóide mapeia os valores de z para que fiquem entre 0 e 1.


```{figure} ../aula7/images/fig4.1.png
---
width: 100%
name: sigmoide
align: center
---
Função sigmóide. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`.)
```

Com a função sigmóide, conseguimos computar a probabilidade $Pr(y=1|X)$. No entanto, precisamos também definir um nível de decisão. Para uma probabilidade de 0.51, classificaremos a observação como da classe 1 ou da classe 0? Apesar de bem próximo ao meio termo, poderíamos adotar o nível de decisão de 0.5, o que jogaria essa observação para classe 1. Isso é o que é conhecido no aprendizado de máquina como *Decision Boundary*.

$$
\operatorname{decision}(x)=
\begin{cases}
1 & \text{se } P\!\bigl(y=1 \mid x\bigr) > 0.5 \\
0 & \text{caso contrário}
\end{cases}
$$

Resumindo, o processo de classificação com a regressão logística é o seguinte: Estimamos um valor $z$ utilizando os pesos e termo de viés estimados pelo classificador durante o treinamento. Passamos cada z pela função sigmóide, mapeando-os para um valor no intervalo [0,1]. Com um nível de decisão definidos pelo pesquisador (ou o padrão do classificador), pegamos a probabilidade estimada e classificamos cada observação de teste entre as classes 0 ou 1.


## Classificação com Regressão Multinomial

Quando temos un número de classes $K>2$, não podemos utilizar a regressão logística ou a função sigmóide. Portanto, para o caso da regressão multinomial, utilizamos outra função que permite mapear os valores estimados $z$ para probabilidades que podem ser usadas para decidir a classe de uma observação, a função ***Softmax***.

### Função Softmax

A função **Softmax** é uma generalização da função Sigmóide, usada no modelo de regressão multinomial para calcular a probabilidade de que as observações pertencem a uma classe k $Pr(Y = k | X)$. A função *softmax* pegar um vetor $\mathbf{z} = [z_1, z_2, ..., z_k]$ das probabilidades de cada classe para $k$ classes e mapeia cada uma para uma distribuição de probabilidades, dentro do intervalo [0,1] e somando 1 ao total. Para um vetor $\mathbf{z}$ de dimensionalidade K, o softmax é definido como:

$$
\operatorname{softmax}(z_i)
   \;=\;
   \frac{\exp\!\bigl(z_i\bigr)}
        {\displaystyle\sum_{j=1}^{K}\exp\!\bigl(z_j\bigr)}
   \quad\text{para } 1 \le i \le K
$$

E o *softmax* para de um vetor de entrada $\mathbf{z}$ também será ele mesmo um vetor:

$$
\operatorname{softmax}(\mathbf{z})
   \;=\;
   \Biggl[
      \frac{\exp\!\bigl(z_1\bigr)}{\displaystyle\sum_{i=1}^{K}\exp\!\bigl(z_i\bigr)},
      \;
      \frac{\exp\!\bigl(z_2\bigr)}{\displaystyle\sum_{i=1}^{K}\exp\!\bigl(z_i\bigr)},
      \; \dots,\;
      \frac{\exp\!\bigl(z_K\bigr)}{\displaystyle\sum_{i=1}^{K}\exp\!\bigl(z_i\bigr)}
   \Biggr]
$$

Por meio dessa função, podemos estimar as probabilidades de que uma observação pertença a cada uma das classes num contexto de $K$ classes. Por exemplo, poderíamos ter o seguinte softmax para uma observação de teste em uma classificação com 6 classes:

$$
[0.05, 0.09, 0.01, 0.1, 0.74, 0.01]
$$

Nessa situação temos que a probabilidade de que a observação seja classificada com da classe de número 5 é de 74% (0.74), e sabemos também os valores estimados para todas as outras classes (e.g. classe 1 = 0.05, etc).



## O aprendizado na Regressão Logística

Como os parâmetros do modelo logístico, os pesos $\mathbf{w}$ e o viés b, são estimados? Precisamos de dois componentes principais para essa estimativa: Uma métrica do quão distante os rótulos previstos $\hat{y}$ estão do valor verdadeiro de $y$. Essa distância é mensurada por meio de uma **função de perda** (*Loss/Cost Function*). Um segundo componente é um **algoritmo de otimização** para atualizarmos os pesos iterativamente para reduzir a função de perda ao máximo. Um algoritmo comumente usado é o *Gradient Descent*. Veremos agora uma função de perda comum (*Cross-Entropy*) e o *Gradient Descent*.

### A função de Perda *Cross-entropy*


```{video} https://www.youtube.com/embed/6ArSys5qHAU?si=qVH8W4n4xPbCNmMX
```





## Notas


