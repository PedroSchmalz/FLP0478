# Treinando uma Rede Neural

Na última seção discutimos a estrutura de uma Rede Neural e sua origem, além de apresentar algumas arquiteturas comuns para classificação de imagens (CNN) e para classificação/previsão de dados sequenciais (RNN). Agora, vamos discutir conceitos importantes para o treinamento de uma RNN, e como eles podem afetar a performance dos diferentes modelos e arquiteturas.

Treinar uma Rede Neural pode ser um pouco complexo, mas entender como funciona esse treinamento pode ser útil para o ajuste de hiperparâmetros e construção de arquiteturas de Redes, além de melhorar a performance do modelo para sua tarefa específica. Vamos voltar em uma figura que vimos na seção anterior.



```{figure} ../aula12/images/islfig10.1.png
---
width: 100%
name: neuralnet2
align: center
---
Rede Neural com uma única Camada Escondida. Fonte: James et al. ({cite}`james2023introduction`., p.400))
```

A {numref}`Figura {number} <neuralnet2>` mostra um modelo cujos parâmetros são $\beta = (\beta_0, \beta_1, ..., \beta_k)$


$$
f(\mathbf{X}) = \beta_0 + \sum_{k=1}^{K} \beta_k h_k(\mathbf{X})
$$
$$
\phantom{f(\mathbf{X})} = \beta_0 + \sum_{k=1}^{K} \beta_k g\Big(w_{k0} + \sum_{j=1}^{p} w_{kj} X_j\Big).
$$

Primeiro, as k ativações $A_k, k=1, ..., K$ na camada escondida são computadas como funções das *features*, ou *inputs*, $x_1,...,x_p$

$$
A_k = h_k(\mathbf{X}) = g\left( w_{k0} + \sum_{j=1}^p w_{kj} X_j \right)
$$

Onde $g(z)$ é uma função de ativação não linear. Podemos pensar em cada $A_k$ como uma transformação $h_k(X)$ diferente das variáveis originais. Todas essas transformações são estão alimentadas na camada de saída, resultando em:

$$
f(\mathbf{X}) = \beta_0 + \sum_{k=1}^K \beta_k A_k
$$

Que é um modelo de Regressão Linear nas $K=5$ ativações. Todos os parâmetros precisam ser estimados a partir dos dados. Nas primeiras aplicações de Redes Neurais, a função sigmóide era favorecida como função de ativação:

$$
g(z) = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}
$$


A mesma utilizada na regressão logística para converter as probabilidades da função linear. Hoje em dia, uma mais comum é a ReLu (*Rectified Linear Unit*):


$$
g(z) = (z)_+ =
\begin{cases}
0 & \text{se } z < 0 \\
z & \text{caso contrário}
\end{cases}
$$

A ReLU pode ser computada e armazenada de forma mais eficiente. No entanto, como mostrado na {numref}`figura {number} <activfunc>`, existem múltiplas funções de ativação que podem ser utilizadas (e sim, isso se torna um hiperparâmetro). O Nome Rede Neural dessa arquitetura mostrada até o momento deriva do fato de que há uma analogia com os neurônios de nosso cérebro. Os valores de ativação $A_k = h_k(X)$ mais perto de 1 vão "ativar" o neurônio, e valores mais perto de 0 não o "ativam".



### Redes Neurais Multicamadas

```{video} https://www.youtube.com/embed/CqOfi41LfDw?si=KsXCu-DOSQiFHUEi
```

Os modelos de Redes Neurais Contemporâneos geralmente têm mais de uma camada escondida. 


```{figure} ../aula12/images/islfig10.4.png
---
width: 100%
name: neuralnetmult
align: center
---
Rede Neural com duas camadas escondidas. Fonte: James et al. ({cite}`james2023introduction`., p.404))
```

A {numref}`Figura {number} <neuralnetmult>` mostra uma Rede Neural com duas camadas escondidas ($L_1$ e $L_2$), e a tarefa é de classificação com dez classes ($Y_0,...,Y_9$). A primeira camada $L_1$ é da mesma forma que vimos antes.


$$
A_k^{(1)} = h_k^{(1)}(\mathbf{X}) = g\left(w_{k0}^{(1)} + \sum_{j=1}^{p} w_{kj}^{(1)} X_j\right)
$$

Para $k=1,...,K$. A segunda camada $L_2$ trata as ativações $A_k^{(1)}$ da primeira camada como valores de entrada (*inputs*) e calcula novas ativações:

$$
A_{\ell}^{(2)} = h_{\ell}^{(2)}(\mathbf{X}) = g\left(w_{\ell 0}^{(2)} + \sum_{k=1}^{K_1} w_{\ell k}^{(2)} A_k^{(1)}\right)
$$

Para $l=1,...,K_2$. Dessa maneira, por meio de uma série de transformações, a RN é capaz de construir transformações complexas de $X$ que são alimentadas na camada de saída como *features*. E como isso acontece? No caso da classificação, o primeiro passo é computar diferentes modelos lineares para cada categoria:


$$
Z_m = \beta_{m0} + \sum_{\ell=1}^{K_2} \beta_{m\ell} h_{\ell}^{(2)}(\mathbf{X})
$$
$$
\phantom{Z_m} = \beta_{m0} + \sum_{\ell=1}^{K_2} \beta_{m\ell} A_{\ell}^{(2)}
$$

O segundo passo, na classificação, é o de transformar cada $Z$ em probabilidades de classe $f_m(X) = Pr(y=m|X)$. Para isso, podemos usar a função de ativação ***softmax***


$$
f_m(\mathbf{X}) = \Pr(Y = m \mid \mathbf{X}) = \frac{e^{Z_m}}{\sum_{\ell=0}^{9} e^{Z_\ell}}
$$

Com essa função, garantimos que, com $m=0,1,...,9$, os números se comportem como probabilidades (não negativas e de soma 1). Mesmo que o objetivo seja de classificação, o modelo irá estimar probabilidades para cada uma das classes. A classificação final de cada observação se dará pela classe com maior probabilidade. A **função de perda/objetiva** irá guiar o treinamento desse modelo, e geralmente é a *negative multinomial log-likelihood*, ou mais conhecida como ***cross-entropy***.

$$
-\sum_{i=1}^{n} \sum_{m=0}^{9} y_{im} \log \left( f_m(x_i) \right)
$$


## Redes Neurais Convolucionais (CNNs)



```{video} https://www.youtube.com/embed/HGwBXDKFk9I?si=v5WSgRDcJtabI38_
```


Antes de partir para a parte técnica de como treinar um modelo de Rede Neural e os principais hiperparâmetros e conceitos asosciados ao treinamento, vamos apresentar duas estruturas muito comumente usadas para dois objetivos diferentes. A primeira estrutura é a ***Convolutional Neural Network***, ou **Redes Neurais Convolucionais**. As *CNNs* são modelos de aprendizado profundo que se destacaram na classificação de imagens. Elas imitam em algum grau a maneira com que humanos classificam imagens, reconhecendo aspectos ou padrões na imagem que permitam distinguir cada classe de objeto.


```{figure} ../aula12/images/islfig10.6.png
---
width: 100%
name: neuralcnn
align: center
---
Ilustração de como uma *CNN* classifica imagens. Fonte: James et al. ({cite}`james2023introduction`., p.404))
```

Na {numref}`Figura {number} <neuralcnn>`, podemos ver como funciona esse processo. A Rede primeiro identifica aspectos de "baixo nível", como bordas, borras de cores, etc. Depois, esses aspectos são combinados para formar aspectos de "alto nível", como pedaços de orelhas, olhos, etc. Eventualmente, a presença ou ausência desses aspectos vão contribuir para o cálculo de probabilidade de cada classe. Para construir essa hierarquia, as CNNs utilizam dois tipos de camadas: as **camadas de convolução** (*Convolution*), que procuram padrões pequenos na imagem, e camadas de **agrupamento** (*pooling*), que reduzem a resolução desses padrões para escolher algum subconjunto proeminente.

### Camadas de Convolução.


A camada de convolução (*convolution*) é feita de filtros de convolução, cada um deles sendo um template que determina se dada *feature* local está presente ou não na imagem. Este filtro depende em uma operação simples, chamada convolução, que consiste em multiplicar repetidamente os elementos de uma matriz e somar os resultados. Peguemos a seguinte "imagem" original:

Imagem Original

$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & i \\
j & k & l
\end{bmatrix}
$$

E o seguinte filtro de Convolução:

$$
\begin{bmatrix}
\alpha & \beta \\
\gamma & \delta
\end{bmatrix}
$$

Isso gerará uma imagem "convolucionada":

$$
\begin{bmatrix}
a\alpha + b\beta + d\gamma + e\delta & b\alpha + c\beta + e\gamma + f\delta \\
d\alpha + e\beta + g\gamma + h\delta & e\alpha + f\beta + h\gamma + i\delta \\
g\alpha + h\beta + j\gamma + k\delta & h\alpha + i\beta + k\gamma + l\delta
\end{bmatrix}
$$

Na prática, teremos algo mais ou menos assim:

```{figure} ../aula12/images/islfig10.7.png
---
width: 100%
name: convolfilter
align: center
---
Imagem passada por um filtro de convolução. Fonte: James et al. ({cite}`james2023introduction`., p.406))
```

### Camada de Agrupamento (*Pooling*)

Depois de passar pela camada de convolução, a imagem será "reagrupada" e condensada em uma imagem menor de resumo. Na arquitetura final de uma CNN, esse processo de convolução e depois *pooling* será repetido algumas vezes até chegar na camada final de saída, que gerará uma classificação daquela imagem.


```{figure} ../aula12/images/islfig10.8.png
---
width: 100%
name: arquitetcnn
align: center
---
Arquitetura de um CNN. Fonte: James et al. ({cite}`james2023introduction`., p.406))
```


## Redes Neurais Recorrentes (RNNs)


```{video} https://www.youtube.com/embed/AsNTP8Kwu80?si=R8mVb6HlHJyUByhU
```


Muitos tipos de dados são sequenciais por natureza, e por isso necessitam de tratamento especial ao se construir modelos preditivos. Alguns exemplos incluem:

* Documentos de Texto
* Séries Temporais (PIB por ano, IDH, Temperatura, etc.)
* Discursos e outras gravações de som
* Escrita à mão.

Em uma **Rede Neural Recorrente**, ou ***Recurrent Neural Network*** (RNN), a entrada $X$ é uma sequência. No caso de um córpus de documentos, cada documento pode ser representado como uma sequência de $L$ palavras, então $X = {X_1,X_2,...,X_L}$, onde cada $X_l$ representa uma palavra (em BOW). A ordem das palavras e sua proximidade em uma frase transmitem significado semântico. os RNNs são desenhados para acomodar e se aproveitar dessa natureza sequencial desse tipo de objeto de entrada. 

```{figure} ../aula12/images/islfig10.12.png
---
width: 100%
name: arquitetrnn
align: center
---
Arquitetura de um RNN. Fonte: James et al. ({cite}`james2023introduction`., p.417))
```

Na {numref}`figura {number} <arquitetrnn>` está ilustrada a estrutura de um RNN básico com uma sequência $X= {X_1,X_2,...,X_L}$ como entrada, uma saída simple $Y$ e uma camada escondida $\{A_{\ell}\}_{\ell=1}^{L} = \{A_{1},\, A_{2},\, \ldots,\, A_{L}\}$. Cada $X_l$ é um vetor. Enquanto a sequência é processada um vetor $X_l$ por vez, a Rede atualiza as ativações $A_l$ na camada escondida, tomando como entrada o vetor $X_l$ e o vetor de ativação $A_{l-1}$ do passo anterior. Esse processo se repete até chegar em $O_L$, o último e mais relevante para o resultado final de predição/classificação (Se o objetivo for prever a próxima palavra, ele não é mais tão relevante). 

Em detalhe, suponha que cada vetor $X_l$ da sequência de entrada tem $p$ componentes $\mathbf{X}_{\ell}^{T} = (X_{\ell 1},\, X_{\ell 2},\, \ldots,\, X_{\ell p})$ e a camada escondida consiste em $K$ unidades $\mathbf{A}_{\ell}^{T} = (A_{\ell 1},\, A_{\ell 2},\, \ldots,\, A_{\ell K})$. Como na figura, representamos a coleção de $K * (p+1)$ pesos compartilhados $w_{kj}$ para a camada de entrada por uma matriz $\mathbf{W}$, e $\mathbf{U}$ é uma matriz de $K*K$ dos pesos $w_k$ para as camadas *hidden-to-hidden*, e $\mathbf{B}$ é um vetor de k+1 pesos $B_K$ para a camada de saída. Então:

$$
A_{\ell k} = g\left(w_{k0} + \sum_{j=1}^p w_{kj} X_{\ell j} + \sum_{s=1}^K u_{ks} A_{\ell-1,\,s}\right)
$$

E a camada de saída $O_l$ é calculada como

$$
O_{\ell} = \beta_{0} + \sum_{k=1}^{K} \beta_{k} A_{\ell k}
$$

Para uma saída quantitativa. Se for uma classificação, uma camada adicional de ativação é utilizada para adequar os resultados de saída.



## Conclusão

A transição para o aprendizado profundo trouxe uma revolução ao permitir que os modelos extraíssem automaticamente representações úteis dos dados, superando a dependência de engenharia manual de features. Nesta aula, mergulhamos nas estruturas e fundamentos das redes neurais, desde unidades computacionais básicas até arquiteturas multicamadas, abordando funções de ativação, modelos de classificação, regressão e estruturas especializadas como CNNs e RNNs, cada qual adaptada para desafios específicos como imagens e sequências. Percebemos como estas arquiteturas conseguem modelar relações cada vez mais complexas e extrair padrões ricos do próprio dado, ampliando consideravelmente o escopo e a precisão das aplicações modernas de machine learning.

Agora vamos discutir como funciona o treinamento de uma Rede Neural: veremos como preparar os dados, definir a função de perda, ajustar os pesos das conexões usando algoritmos como backpropagation, e como hiperparâmetros como taxa de aprendizado, épocas e tamanho de lote influenciam o sucesso do modelo. Explicaremos também como avaliar o desempenho e evitar problemas de overfitting, preparando a base para sua aplicação prática com ferramentas como PyTorch.




