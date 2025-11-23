# Introdução ao Aprendizado Profundo

Na última aula vimos os fundamentos e vantagens dos métodos ensemble, com destaque para Random Forests e outras técnicas de combinação de modelos. Estudamos como o Bagging (Bootstrap Aggregating) combina diversas árvores de decisão, treinadas em amostras diferentes do conjunto de dados, para reduzir a variância e aumentar a robustez das previsões. Observamos que o Random Forest aprimora essa ideia ao sortear aleatoriamente preditores em cada divisão das árvores, tornando os modelos base menos correlacionados e melhorando a performance geral. Exploramos ainda o conceito de métricas de importância de preditores, permitindo recuperar interpretabilidade ao identificar quais variáveis são mais relevantes dentro de um ensemble. Além disso, discutimos o Boosting, que constrói modelos de forma sequencial corrigindo os erros anteriores, e técnicas avançadas como AdaBoost, Gradient Boosting e XGBoost. Por fim, aprendemos sobre métodos que combinam diferentes algoritmos básicos, como Voting Classifiers e Stacking, aumentando ainda mais o poder preditivo ao integrar a diversidade de abordagens em uma solução única e eficiente.


Nesta aula vamos nos aprofundar no universo do ***Deep Learning*** (**Aprendizado Profundo**), explorando como funcionam e são treinados os modelos de redes neurais artificiais. Vamos entender o ciclo completo de treinamento desses modelos: desde a definição das funções de perda e o uso da técnica de backpropagation para ajustar os pesos das conexões, passando pelo conceito de épocas, batches e iterações, até práticas de validação e otimização dos hiperparâmetros. Além disso, faremos uma introdução prática ao PyTorch, uma das bibliotecas mais populares e flexíveis para desenvolvimento de deep learning atualmente. O PyTorch se destaca tanto pelo seu modelo de computação dinâmica, que facilita a construção e o diagnóstico de modelos complexos, quanto pela forte adoção na comunidade acadêmica e incansável foco em experimentação e prototipagem científica — motivos que o tornam a escolha ideal para quem deseja entender, implementar e inovar em aplicações modernas de redes neurais profundas.

Até a última década, os modelos de Aprendizado de Máquina dependiam fortemente de engenharia de *features*, que consiste em chegar nas transformações corretas dos dados para que o modelo conseguisse resolver as tarefas e problemas da melhor forma. O ***Deep Learning***, ou **Aprendizado Profundo**, consegue achar essas transformações/representações automaticamente, usando o dado em sua forma bruta. A habilidade destes modelos de ingerir dados e extrair representações úteis é o que faz com que o *Deep Learning* seja tão poderoso. O pilar que sustenta essa área da Inteligência Artificial é a **Rede Neural** (*Neural Network*). Hoje conheceremos o que é a Rede Neural e como aplicá-la na classificação de texto.


## Rede Neural


```{video} https://www.youtube.com/embed/aircAruvnKk?si=10baygyjCfA9JWz9
```

As **Redes Neurais** (***Neural Networks***) tiveram um primeiro momento de fama nos anos 80. Mas, devido ao seu alto custo computacional e ncessidade de ajuste fino de hiperparâmetros, foram deixadas de lado em favor de modelos mais simples e que performavam melhor. Porém, com o aumento da disponibilidade de bancos de dados maiores (*Big Data*) e a possibilidade de paralelizar o processamento por meio de *GPUs* (Placas de vídeo/Gráficas), veio a "Renascença" das Redes Neurais, e a área voltou a ganhar força. Hoje em dia, as Redes Neurais são uma ferramenta essencial para o Processamento de Língua Natural. Apesar deste momento de fama nos anos 80, a origem do nome vem de um modelo de Mcculocch-Pitts em 1943, que buscava simplificar o neurônio biológico como um elemento de computação que poderia ser descrito em termos de lógica propositiva ({cite}`mcculloch1943logical`., 1943). O uso moderno se distanciou bastante disso. Hoje, a Rede Neural é uma rede de *unidades* computacionais menores ({cite}`jurafsky2024speech`.), e cada uma delas pegará um vetor de valores de entrada ($\mathbf{X}$) e produzirá um único valor de saída $y$. A primeira arquitetura introduzida é a ***Feedforward Network*** (Rede Progressiva), que possui esse nome pois a computação é iterativa de uma camada para a próxima. O uso contemporâneo se chama aprendizado profundo por que as Redes Neurais atuais possuem múltiplas camadas.


### Unidade (*Unit*)

O elemento básico de uma Rede Neural é uma única unidade computacional. A unidade toma um conjunto de valores reais como entrada, faz operações e produz uma saída. De forma geral, a unidade pega uma soma ponderada das entradas, com um termo adicional que é o termo de viés. Dado um conjunto de entradas (*inputs*) $X_1, X_2, ..., X_n$, a unidade terá um conjunto de pesos $w_1, w_2, ..., w_n$ e um termo de viés $b$, então a soma ponderada $z$ pode ser representada como:


$$
z = b + \sum_{i} w_i x_i
$$


Em notação matricial, temos:

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

No entanto, ao invés de modelar z como uma função linear de x, as unidades neurais aplicam uma função não linear $f()$, permitindo transformações mais complexas dos dados. A {numref}`Figura {number} <unitneural>` mostra um esquema básico de uma unidade neural. Neste exemplo, a unidade toma 3 valores de entrada $x_1,x_2,x_3$ e calcula uma soma ponderada, multiplicando cada valor por um peso $w_1,w_2,w_3$, e somando eles ao termo de viés $b$, e passando essa soma resultante por uma função de ativação (no exemplo, a função sigmóide, que já vimos antes), resultando em um número entre 0 e 1.


```{figure} ../aula12/images/jurfig6.2.png
---
width: 100%
name: unitneural
align: center
---
Unidade Neural. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 121).
```

Vamos calcular um exemplo manualmente só para termos a intuição por trás dos calculos. Considere uma unidade com o seguinte vetor de pesos e bias:

$$
\mathbf{w} = [0.2,\,0.3,\,0.9]
$$
$$
b = 0.5
$$
O que essa unidade faz com o seguinte vetor de entrada?

$$
\mathbf{x} = [0.5,\,0.6,\,0.1]
$$

O resultado da saída \(y\) será:

$$
y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)
= \frac{1}{1 + e^{-\left(\mathbf{w} \cdot \mathbf{x} + b\right)}}
= \frac{1}{1 + e^{-(0.5 * 0.2 + 0.6 * 0.3 + 0.1 * 0.9 + 0.5)}}
= \frac{1}{1 + e^{-0.87}}
\approx 0.70
$$

Na prática a função sigmóide não é mais tão utilizada, e outras funções podem ser usadas para apropriar o valor final, como a ReLU e a tanh. Aqui estão alguns exemplos de funções que podem ser utilizadas para além da sigmóide:


```{figure} ../aula12/images/stevensfig6.5.png
---
width: 100%
name: activfunc
align: center
---
Funções de ativação comuns. Fonte: Stevens, Antiga e Viehmann, 2020. ({cite}`stevens2020deep`., p. 147).
```


### Rede Neural de Camada Única

Uma Rede Neural será composta de Unidades Neurais como a vista acima. O modelo mais simples de Rede Neural é a **Rede Neural de Camada Única**. Uma Rede Neural vai pegar os *inputs* de $p$ variáveis $X = (x_1,x_2,...,x_p)$ e vai construir uma função não linear $f(x)$ para prever a resposta $y$. Isso não é novo, e já rodamos outros modelos clássicos que faziam isso (e.g. Árvores de decisão, *boosting*, etc). O que distingue as Redes Neurais é sua estrutura.


```{figure} ../aula12/images/islfig10.1.png
---
width: 100%
name: neuralnet
align: center
---
Rede Neural com uma única Camada Escondida. Fonte: James et al. ({cite}`james2023introduction`., p.400))
```

A {numref}`Figura {number} <neuralnet>` mostra uma Rede Neural Progressiva (Pois as informações vão para as camadas seguintes e não há ciclos) com uma única camada escondida. Os *inputs* $X_1$ a $X_4$ são passados para cada uma das unidades de ativação $A_1,...,A_5$, e em cada uma delas o cálculo que mostramos na seção anterior é feito. Essa camada escondida, ou camada de neurônios de ativação, então vai passar esses valores calculados para a camada de output, gerando um único valor predito. Neste exemplo, como não é uma classificação, não existe uma camada de ativação modificando os resultados para ficar entre um intervalo específico. Formalizado, o modelo é da seguinte forma:


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









## Conclusão


Nesta aula exploramos os métodos ensemble que transformam a ideia simples de "a sabedoria das multidões" em ferramentas computacionais sofisticadas capazes de superar significativamente modelos individuais. Começamos com o *Bagging*, que demonstra como treinar múltiplos modelos independentes em subconjuntos aleatórios dos dados reduz a variância através de agregação paralela, e vimos como o *Random Forest* aprimorou essa ideia ao decorrelacionar as árvores através de amostragem aleatória de preditores. Transitamos então para o *Boosting*, uma abordagem fundamentalmente diferente que reduz o viés através do aprendizado sequencial, começando com AdaBoost, evoluindo para a sofisticação do Gradient Boosting, e finalmente chegando em implementações otimizadas como XGBoost, que domina competições reais de ciência de dados através de paralelização, regularização e manejo inteligente de dados ausentes.

Com isso, finalizamos os principais modelos e métodos antes de *deep learning*. Agora, veremos por que esses modelos são chamados de *deep* e por que acabaram substituindo os modelos clássicos para as aplicações de aprendizado de máquina. No entanto, com isso vem o custo de menor interpretabilidade, e também maior custo computacional.




