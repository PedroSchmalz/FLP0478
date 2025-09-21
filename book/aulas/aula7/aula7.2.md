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

Resumindo, o processo de classificação com a regressão logística é o seguinte: Estimamos um valor $z$ utilizando os pesos e termo de viés estimados pelo classificador durante o treinamento. Passamos cada z pela função sigmóide, mapeando-os para um valor no intervalo [0,1]. Com um nível de decisão definidos pelo pesquisador (ou o padrão do classificador), pegamos a probabilidade estimada e classificamos cada observação de teste entre as classes 0 ou 1. A {numref}`Figura {number} <logclass>` mostra como a regressão logística calcula a probabilidade de uma observação ser de uma classe dada as *features* de texto.



```{figure} ../aula7/images/fig4.3.1.png
---
width: 100%
name: logclass
align: center
---
Classificação na regressão Logística Binária. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`.)
```

## Classificação com Regressão Multinomial

Quando temos un número de classes $K>2$, não podemos utilizar a regressão logística ou a função sigmóide. Portanto, para o caso da regressão multinomial, utilizamos outra função que permite mapear os valores estimados $z$ para probabilidades que podem ser usadas para decidir a classe de uma observação, a função ***Softmax***.

### Função Softmax



```{video} https://www.youtube.com/embed/KpKog-L9veg?si=my4iOKA4GkFuMT6U
```

---

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

Nessa situação temos que a probabilidade de que a observação seja classificada com da classe de número 5 é de 74% (0.74), e sabemos também os valores estimados para todas as outras classes (e.g. classe 1 = 0.05, etc). A {numref}`Figura {number} <multclass>` mostra como a regressão multinomial calcula as probabilidades de uma observação ser de uma classe dada as *features* de texto.



```{figure} ../aula7/images/fig4.3.2.png
---
width: 100%
name: multclass
align: center
---
Classificação na regressão Multinomial. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`.)
```




## O aprendizado na Regressão Logística/Multinomial

Como os parâmetros do modelo logístico, os pesos $\mathbf{w}$ e o viés b, são estimados? Precisamos de dois componentes principais para essa estimativa: Uma métrica do quão distante os rótulos previstos $\hat{y}$ estão do valor verdadeiro de $y$. Essa distância é mensurada por meio de uma **função de perda** (*Loss/Cost Function*). Um segundo componente é um **algoritmo de otimização** para atualizarmos os pesos iterativamente para reduzir a função de perda ao máximo. Um algoritmo comumente usado é o *Gradient Descent*. Veremos agora uma função de perda comum (*Cross-Entropy*) e o *Gradient Descent*.

### A função de Perda *Cross-entropy*


```{video} https://www.youtube.com/embed/6ArSys5qHAU?si=qVH8W4n4xPbCNmMX
```
--- 

Precisamos de uma função de perda que expresse, para uma observação, o quão próximo uma saída do classificador $\hat{y} = \mathbf{\sigma} (\mathbf{w}*\mathbf{x} + b) $ é em relação ao valor correto de base, $Y$. Formalizando, queremos descobrir

$$
L(\hat{y},y)
$$

Que é quanto $\hat{y}$ difere de $y$. Fazemos isso por meio de uma funçaõ de perda que vai preferir que a classificação correta durante o treinamento seja a mais provável. Isso é conhecido como *MLE* condicional: escolhemos os parâmetros $w,b$ que maximizam a probabilidade dos valores verdadeiros de y dados os valores x de treinamento. A função de perda resultate é a *negative log likelihood loss*, mais conhecida como ***Cross-Entropy Loss***.


### Descida do Gradiente

```{video} https://www.youtube.com/embed/IHZwWFHWa-w?si=FMtTFGir0x5qFyCX
```
---

O objetivo da descida do gradiente (*Gradient Descent*) é achar os pesos ótimos, minimizando nossa função de perda para todas as observações. Como podemos achar o mínimo dessa função? O método da descida do gradiente procura o mínimo da função descobrindo em qual direção (dentro do espaço de parâmetros) a inclinação da função está aumentando mais rapidamente, e então se move na direção contrária. No caso da regressão logística (ou multinomial), estamos navegando um espaço de parâmetros 2D: Temos o peso $w$ e navegamos ele ao longo da função de perda. Em modelos mais complexos, como os de redes neurais, o espaço de parâmetros passa a ter mais dimensões.  A {numref}`Figura {number} <gradientdesc>` mostra como o algortimo de otimização percorre a função de perda para tentar encontrar o mínimo global, reduzindo a função de perda e garantindo melhores previsões do modelo de aprendizado supervisionado.


```{figure} ../aula7/images/fig4.5.png
---
width: 100%
name: gradientdesc
align: center
---
Movimentação do Algoritmo de Otimização. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`.)
```

O algoritmo de descida do gradiente procura o gradiente da função de perda no ponto atual e se move na direção contrária. Podemos pensar no gradiente como a inclinação da reta (reta pontilhada verde na figura), na situação de uma única variável de otimização. A velocidade desse "passo" da descida do gradiente será ditada por um hiperparâmetro[^1], a **Taxa de Aprendizado**.

### Taxa de Aprendizado

Quanto o algoritmo de otimização vai se "mexer" a cada iteração da descida do gradiente vai depender da **Taxa de Aprendizado**, ou *Learning Rate*. Uma alta taxa de aprendizado faz com que o algoritmo de um "passo" maior, se movendo mais ao longo da curva de aprendizado (Loss x Weights). Isso faz com que o modelo treine mais rapidamente mas, em contextos de aprendizado profundo, pode ser que ele fique em um mínimo local, não minimizando o erro ao máximo. Inversamente, uma taxa de aprendizado baixa faz com que o modelo precise de mais iterações para definir um mínimo, mas não garante que ele vai chegar no mínimo global.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"A taxa de aprendizado η é um hiperparâmetro que precisa ser ajustado. Se for muito alta, o modelo dará passos excessivamente grandes e ultrapassará o mínimo da função de perda; se for muito baixa, dará passos muito pequenos e demorará para chegar ao mínimo. É comum começar com uma taxa de aprendizado maior e reduzi-la gradualmente, fazendo-a variar com a iteração k do treinamento."
({cite}`jurafsky2024speech`., p. 77, tradução nossa)
```




## Notas

[^1]: Hiperparâmetro é uma variável de configuração definida manualmente antes do treinamento que controla aspectos essenciais de como o algoritmo de machine learning aprende, como a taxa de aprendizado, o número de camadas de uma rede neural ou o tamanho do lote de dados. Diferentemente dos parâmetros do modelo (pesos e vieses), que são ajustados automaticamente pelo processo de otimização, os hiperparâmetros precisam ser escolhidos e refinados pelo pesquisador — prática conhecida como ajuste (ou otimização) de hiperparâmetros — porque influenciam diretamente a velocidade de convergência, a capacidade de generalização e o risco de sobreajuste do modelo.


