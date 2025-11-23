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

A {numref}`Figura {number} <neuralnet2>` mostra um modelo cujos parâmetros são $\beta = (\beta_0, \beta_1, ..., \beta_k)$, assim como temos cada um dos pesos $w_k = (w_{k0}, ... , w_{kp})$, k=1, ..., K. Dadas as observações ($x_i,y_i$), $i = 1, ... , n$, nós podemos ajustar o modelo por meio do seguinte problema não linear.

$$
\text{min}_{\{w_k\}_{1}^{K_1},\, \beta} \quad \frac{1}{2} \sum_{i=1}^{n} (y_i - f(x_i))^2
$$

Onde 


$$
f(x_i) = \beta_0 + \sum_{k=1}^{K} \beta_k g\left( w_{k0} + \sum_{j=1}^{p} w_{kj} x_{ij} \right)
$$

O objetivo desse problema é simples: minimizar a distância entre o valor observado e o valor predito pela função $f(x_i)$. No entanto, por ser um problema não convexo nos parâmetros, existem múltiplas soluções possíveis. Aqui entramos no problema mencionado brevemente nas últimas aulas de mínimos locais e mínimos globais. Para evitar o problema de cair em um mínimo local, ou evitar o overfitting, é necessário adotar duas estratégias:

- 1 - Adotar um processo de treinamento lento, usando a descida do gradiente. Também é importante parar o treinamento quando o *overfitting* for detectado;
- 2 - Adotar estratégias de Regularização, reduzindo o número de parâmetros estimados e usando estratégias de *Dropout*.

Vamos começar primeiro pela descida do gradiente.


## Descida do Gradiente


```{video} https://www.youtube.com/embed/IHZwWFHWa-w?si=xIiY6NWQsStARaCT
```

Suponha que representemos os parâmetros em um único vetor $\mathbf{\theta}$. Podemos reescrever a função objetiva/de perda como:


$$
R(\theta) = \frac{1}{2} \sum_{i=1}^{n} (y_i - f_{\theta}(x_i))^2
$$

Onde $f$ depende dos parâmetros estimados. A ideia por trás da descida do gradiente é simples:

* 1. Começar com um chute $\theta^0$ para todos os parâmetros em $\theta$, em t = 0.
* 2. Repetir o processo até que não haja mais redução na função objetiva/de perda $R(\theta)$.




```{figure} ../aula12/images/islfig10.17.png
---
width: 100%
name: gradientdescent
align: center
---
Descida do gradiente para um $\theta$ unidimensional. Fonte: James et al. ({cite}`james2023introduction`., p.428))
```

A {numref}`Figura {number} <gradientdescent>` mostra uma possível jornada dentro do espaço de valores de $\theta$. Veja que o mínimo global é marcado por $R(\theta^7)$, indicando o menor erro possível dentro desse modelo mais simples com uma única camada escondida (isso não é tão bonito com modelos maiores e com mais camadas). No entanto, há um ponto ali próximo do valor $-0.5$ no eixo de $\theta$ que pode indicar um possível mínimo local em que o modelo poderia ficar preso. Na prática, o $\theta$ começa com um chute e vai ajustando para minimizar o erro, mas o 'passo' dado pelo modelo é pequeno, e pode ser que ficaríamos presos neste ponto em uma determinada iteração, não atingindo o menor erro possível.

## Retropropagação (*Backpropagation*)


```{video} https://www.youtube.com/embed/Ilg3gGewQ5U?si=SPPGhR6Z0niYtT0d
```

Como a descida do gradiente decide para onde se mover para reduzir a função objetivo? o Gradiente de $R(\theta)$, avaliado em algum ponto $\theta = \theta^m$, é o vetor de derivadas parciais (não se preocupe com isso muito) naquele ponto:

$$
\nabla R(\theta^m) = \left. \frac{\partial R(\theta)}{\partial \theta} \right|_{\theta = \theta^m}
$$

O superscrito $\theta = \theta^m$ significa que depois de calcular o vetor de derivadas, nós o avaliamos no chute atual, $\theta^m$. Isso nos dá uma estimativa da direça no espaço $\theta$ em que $R(\theta)$ aumenta mais rapidamente. A ideia por trás da descida do gradiente é mover o vetor de parâmetros $\theta$ um **pouco** na direção contrária (lembre-se que o objetivo da função de perda é minimizar a perda):

$$
\theta^{m+1} \leftarrow \theta^m - \rho \nabla R(\theta^m)
$$

Para um valor pequeno da taxa de aprendizado $\rho$, o passo vai diminuir a função objetivo $R(\theta)$. Se o vetor de gradientes é zero, então chegamos em um mínimo do objetivo (que pode ser o mínimo local).


### Descida do Gradiente Estocástica (*Sthocastic Gradient Descent*, SGD)

A descida do gradiente geralmente leva muitos passos (*steps*) para atingir um mínimo. Existem algumas formas de acelerar o processo. Quando o $n$ é suficientemente grande, podemos amostrar uma pequena fração ou minilote (*minibatch*) de cada vez e calcula um passo do gradiente. Esse é o processo conhecido como *SGD*.


## Regularização e Dropout


Para garantir que uma Rede Neural seja capaz de generalizar para dados novos e não apenas memorizar o conjunto de treinamento, aplicam-se técnicas de regularização e dropout durante o treinamento do modelo. Essas abordagens são fundamentais para evitar o chamado overfitting, quando o modelo aprende os detalhes ou ruídos específicos dos dados de treinamento, tornando-se pouco eficiente em situações reais.​

A regularização tradicional, como L1 e L2, adiciona penalidades à função de perda, incentivando a redução dos pesos dos parâmetros. Isso tem o efeito de simplificar o modelo e impedir que conexões individuais se tornem excessivamente influentes, o que pode prejudicar a capacidade de generalização. Além disso, o chamado decaimento de pesos (uma forma de regularização L2) atua reduzindo a soma dos quadrados dos pesos, induzindo esparsidade e removendo conexões desnecessárias da rede.​

O dropout, por sua vez, é uma técnica dinâmica que modifica a arquitetura da rede durante o próprio treinamento. Em cada iteração do treinamento, uma fração dos neurônios ocultos é desligada de maneira aleatória, e as conexões associadas a esses neurônios também deixam de atuar temporariamente. Dessa forma, cada mini-lote é processado por uma versão ligeiramente diferente da rede, obrigando os neurônios a aprenderem padrões sem depender de conexões específicas. No final do treinamento, ao restaurar todos os neurônios e normalizar os pesos, a rede se torna muito mais robusta e capaz de generalizar para novos dados. A taxa de dropout, geralmente definida entre 20% e 50%, é um hiperparâmetro importante e pode ser combinada com outros métodos de regularização para potencializar a robustez do modelo.​

Em resumo, regularização e dropout trabalham juntos para construir modelos mais simples, estáveis e capazes de encontrar padrões relevantes sem memorizar detalhes irrelevantes dos dados.

## *Tuning* da Rede Neural

Algumas escolhas de hiperpârametros podem afetar a performance de redes neurais:

* Número de camadas escondidas, e número de unidades por camada
* Regularização: Se vai ter *dropout* ou não, força da regularização de Ridge (L1 ou L2);
* Detalhes relativos à descida do gradiente: Tamanho dos *minibatches*, número de epochs (iterações) em que o modelo será treinado, taxa de aprendizado (que interaje com número de epochs), etc.



## Conclusão

Ao final desta seção, fica claro que dominar o treinamento de redes neurais exige atenção cuidadosa à arquitetura dos modelos, à escolha da função de perda, à aplicação estratégica de regularização e ao uso do dropout para evitar sobreajustes. Abordamos os principais fundamentos matemáticos, desde a descida do gradiente até técnicas modernas de ajuste de hiperparâmetros, e mostramos como diferentes estruturas — como CNNs para imagens e RNNs para sequências — podem ser customizadas para tarefas específicas. 

No tutorial da aula de hoje veremos uma introdução ao Pytorch, e como construir uma Rede Neural simples (uma ou poucas camadas) para classificação de texto, ainda utilizando métodos baseados em BOW, como *TF-IDF*. Na próxima aula veremos que são *embeddings* e como são usados em conjunto com *transformers* para gerar modelos de *deep learning* com ótimos resultados.


