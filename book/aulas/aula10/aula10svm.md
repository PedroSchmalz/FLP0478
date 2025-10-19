# Máquinas de Vetores de Suporte

Na seção anterior, trabalhamos o funcionamento do modelo de árvores de decisão. Agora, veremos o segundo modelo da aula, o modelo de **Máquinas de Vetores de Suporte**, ou *Support Vector Machines*


```{video} https://www.youtube.com/embed/efR1C6CvhmE?si=u_uVRbYz4LlsV6PG
```

---

O *Support Vector Machine*, ou Máquina de Vetores de Suporte, é um modelo que surge como generalização de um classificador mais simples, o *maximal margin classifier* (Classificador de margem máxima). Este classificador exigia que houvesse uma separação por meio de um limite linear no espaço de preditores, o que não é possível em muitas situações. Por isso, novos classificadores com base nele surgiram, até chegarmos no *SVM* de fato. Vamos primeiro entender o que é o Classificador de Margem Máxima e como ele funciona.


## Classificador de Margem Máxima

Para entender esse classificador, primeiro precisamos entender o que é um hiperplano.

### O hiperplano

Em um espaço com $p$ dimensões, um hiperplano é um subespaço afim plano de dimensão $p-1$. Por exemplo, em duas dimensões o hiperplano é um subespaço plano unidimensional. Ou seja, uma reta. Em três dimensões, um hiperplano é um subespaço plano bidimensional, um plano. Em $p >3$, a coisa se complica um pouco e fica mais difícil de visualizar, mas a ideia se mantém. Podemos pensar no hiperplano como dividindo um espaço de dimensão $p$ em duas metades.



```{figure} ../aula10/images/islfig9.1.png
---
width: 100%
name: hiperplano
align: center
---
Um hiperplano dividindo o espaço dos preditores $X_1$ e $X_2$ na metade. Fonte: James et al. ({cite}`james2023introduction`., p. 369)
```

A {numref}`Figura {number} <hiperplano>` mostra o hiperplano de equação $ 1 + 2X_1 + 3X_2 = 0$. Valores em que essa equação são maiores que zero estão coloridos de azul, e valores menores que zero estão em roxo/rosa. Como estamos em um espaço de dimensão $p=2$, o hiperplano é uma reta. 



### Classificação usando um hiperplano

É possível pensar a construção de um hiperplano que separa as observações de treinamento de acordo com suas classes. Como visto na figura acima, já temos, só com o hiperplano, a separação entre observações "azuis" e observações "roxas". 

```{figure} ../aula10/images/islfig9.2.1.png
---
width: 100%
name: hiperplano2
align: center
---
Possíveis hiperplanos dividindo o espaço dos preditores $X_1$ e $X_2$ na metade. Fonte: James et al. ({cite}`james2023introduction`., p. 370)
```

A {numref}`Figura {number} <hiperplano2>` mostra algumas retas (hiperplanos) possíveis na separação do espaço de preditores em duas metades. Se existe um hiperplano (nem sempre existe), podemos usá-lo como um classificador "natural": A observação de teste será classificada com base em qual lado do hiperplano ela está, como mostra a figura abaixo:

```{figure} ../aula10/images/islfig9.2.2.png
---
width: 100%
name: hiperplano3
align: center
---
Hiperplano dividindo as observações entre roxos e azuis. Fonte: James et al. ({cite}`james2023introduction`., p. 370)
```


### Qual o melhor hiperplano?

Nem sempre existe um hiperplano que separa as observações. Quando existe, tem mais de uma possibilidade. Precisamos, então, decidir qual hiperplano iremos utilizar para classificar as observações. A escolha mais natural (segundo os autores) é a do **hiperplano de margem máxima**, ou o **hiperplano ótimo de separação**, que é o hiperplano mais distante das observações de treinamento em ambas as classes. Ou seja, podemos calcular a distância perpendicular de cada observação de treinamento até um dado hiperplano de separação: a menor dessas distâncias é a distância mínima das observações até o hiperplano; e é conhecida como **margem**. O hiperplano de margem máxima é o hiperplano de separação para o qual a margem é a maior possível. Isto é, o hiperplano que ter a maior distância mínima das observações de treinamento.

```{figure} ../aula10/images/islfig9.3.png
---
width: 100%
name: hiperplano3
align: center
---
Hiperplano de margem máxima. Fonte: James et al. ({cite}`james2023introduction`., p. 371)
```

Olhando para a {numref}`Figura {number} <hiperplano3>`, vemos que o hiperplano tem uma margem delimitada pela linha pontilhada. Nesse caso, essa é a maior margem possível entre as observações de cada classe que foi encontrada com base nessas variáveis. Podemos ver também que três observações de treinamento são equidistantes do hiperplano (marcadas pelas setas). Essas observações são conhecidas como os **vetores de suporte**, dado que são vetores em um espaço de $p$ dimensões e dão "suporte" ao hiperplano. Isto é, se as observações mudassem, o hiperplano de margem máxima também mudaria. Por isso, ele é muito sensível às observações próximas da divisão, e pouco/nada sensível às observações distantes.


## Classificadores de Vetores de Suporte (*SVC*)

Nem sempre é possível separar as observações com um hiperplano. E mesmo quando é possível, talvez não seja desejável usar limites tão rígidos quanto os necessários para a definição do hiperplano de margem máxima, que separa perfeitamente as observações entre as classes. Por isso, talvez seja útil relaxar essa restrição, aumentando a robustez do modelo às observações dos vetores de suporte, e garantindo maior generalização dos resultados para dados não vistos. Essa é a ideia por trás dos **Classificadores de Vetores de Suporte**, ou *Support Vector Classifiers*, que usam uma margem suave, permitindo que algumas observações estejam dentro da margem, ou do lado contrário dela.


```{figure} ../aula10/images/islfig9.6.png
---
width: 100%
name: hiperplano4
align: center
---
Classificadores de Vetores de Suporte de margem "suave". Fonte: James et al. ({cite}`james2023introduction`., p. 371)
```

Na {numref}`Figura {number} <hiperplano4>`, o hiperplano admite que algumas observações estejam dentro da margem. Além disso, também permite que algumas observações estejam na "arquibancada da torcida rival", como é o caso das observações azuis 1 e 12, e a observação roxa de número 11. Com isso, aumentamos um pouco o viés do modelo, mas garantimos menor variância e menos variação com base nas observações de suporte. 

O quão "suave" essa margem é, é definida com com base em um **Hiperparâmetro**, o $C$, que define quanta violação da margem será tolerada. Quanto menor o C, menos viés o modelo terá (menos flexível). Quanto maior o valor desse parâmetro, maior a flexibilidade.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Agora consideramos o papel do parâmetro de ajuste C. Em (9.15), C limita a soma dos εᵢ, e portanto determina o número e a severidade das violações à margem (e ao hiperplano) que toleraremos. Podemos pensar em C como um **orçamento** para a quantidade de violação da margem que pode ser cometida pelas n observações. Se C = 0, então não há orçamento para violações à margem, e deve ser o caso que ε₁ = ··· = εₙ = 0, caso no qual (9.12)–(9.15) simplesmente se reduz ao problema de otimização do hiperplano de margem máxima (9.9)–(9.11). (É claro, um hiperplano de margem máxima existe apenas se as duas classes forem separáveis.) Para C > 0, não mais que C observações podem estar no lado errado do hiperplano, porque se uma observação está no lado errado do hiperplano então εᵢ > 1, e (9.15) requer que Σⁿᵢ₌₁ εᵢ ≤ C. À medida que o orçamento C aumenta, nos tornamos mais tolerantes às violações da margem, e assim a margem irá alargar. Inversamente, à medida que C diminui, nos tornamos menos tolerantes às violações da margem e assim a margem se estreita.
"
({cite}`james2023introduction`., p. 378, tradução nossa)
```

## Máquinas de Vetores de Suporte (*SVM*)


```{video} https://www.youtube.com/embed/Toet3EiSFcM?si=Ef0hDB6E76GuRvoH
```

---





E quando a separação do espaço de preditores não é linear? 


```{figure} ../aula10/images/islfig9.8.1.png
---
width: 100%
name: svm
align: center
---
Classificadores de Vetores de Suporte de margem "suave". Fonte: James et al. ({cite}`james2023introduction`., p. 371)
```

Na {numref}`Figura {number} <svm>`, não é possível estabelecer um hiperplano que corta exatamente as classes em duas metades. Por isso, utilizamos as Máquinas de Vetores de Suporte, ou *Support Vector Machines*, que lidam com a não linearidade de forma automática, sem precisar saturar o modelo colocandos os polinômios das variáveis preditoras (e.g. $X_1^2$, $X_1^3$, $X_1^4$, e assim por diante). O *SVM* é uma extensão do *SVC* que resulta da saturação do espaço de preditores utilizando de *Kernels* para lidar com a não linearidade.

O kernel (ou núcleo) é uma função matemática que permite ao SVM realizar um "truque" elegante: ao invés de você manualmente criar todas as variáveis polinomiais possíveis para capturar relações não-lineares nos dados originais, o kernel automaticamente transforma os dados para um espaço de dimensão superior onde eles se tornam linearmente separáveis. Imagine que você tem pontos distribuídos em círculos concêntricos em 2D — impossíveis de separar com uma linha reta. O kernel RBF (Radial Basis Function), por exemplo, "projeta" esses pontos para um espaço 3D onde eles podem ser separados por um plano. O mais impressionante é que essa transformação acontece de forma implícita: o algoritmo nunca calcula explicitamente as coordenadas no novo espaço de alta dimensão, apenas calcula produtos internos através da função kernel, tornando o processo computacionalmente eficiente. Os kernels mais comuns são o linear (para dados já separáveis), polinomial (para relações polinomiais), RBF/Gaussiano (para fronteiras complexas e curvas), e sigmoide (similar a redes neurais). A escolha do kernel e seus parâmetros (como o grau do polinômio ou o gamma do RBF) são hiperparâmetros cruciais que devem ser ajustados usando técnicas como grid search ou random search para otimizar a performance do modelo.

```{figure} ../aula10/images/islfig9.9.png
---
width: 100%
name: svmkernel
align: center
---
Esquerda: Um SVM com kernel polinomial de grau 3 é aplicado aos dados não lineares da Figura 9.8, resultando em uma regra de decisão muito mais apropriada. Direita: Um SVM com kernel radial é aplicado. Neste exemplo, qualquer um dos kernels é capaz de capturar a fronteira de decisão. Fonte: James et al. ({cite}`james2023introduction`., p. 371)
```

na {numref}`Figura {number} <svmkernel>` temos os mesmos dados da figura anterior. Na figura da esquerda, utiliza-se um kernel polinomial de grau 3 para ajustar melhor às observações, comportando a não linearidade. Na figura da direita, também lidamos com a não linearidade, mas usando de um kernel radial. Lembre-se de que isso sempre cai no trade-off de flexibilidade: Modelos mais flexíveis são mais propensos ao *overfitting*. A escolha do *kernel* em si se torna um hiperparâmetro, que deve ser escolhido com base em validação cruzada.

## Conclusão


Nesta seção exploramos as Máquinas de Vetores de Suporte (SVMs), um dos algoritmos mais robustos e versáteis do aprendizado de máquina, desenvolvido por Vladimir Vapnik e seus colegas na década de 1990. Compreendemos que os SVMs surgem como uma generalização progressiva de classificadores mais simples: começando pelo Classificador de Margem Máxima, que encontra o hiperplano que maximiza a distância (margem) entre as classes quando os dados são perfeitamente separáveis linearmente, passando pelo Classificador de Vetores de Suporte (SVC), que introduz a margem "suave" através do hiperparâmetro C para permitir violações e aumentar a robustez do modelo, até chegar finalmente ao SVM completo, que utiliza o kernel trick para lidar elegantemente com relações não-lineares sem a necessidade de criar manualmente termos polinomiais. O conceito de vetores de suporte é central nesse algoritmo: apenas os pontos de dados mais próximos do hiperplano (os que "tocam" as fronteiras da margem) definem a solução, tornando o método eficiente em termos de memória e computacionalmente elegante. O hiperparâmetro C funciona como um orçamento que controla o trade-off entre maximizar a margem e minimizar erros de classificação: valores pequenos de C resultam em margens mais largas mas mais tolerantes a erros (maior viés, menor variância), enquanto valores grandes de C buscam classificar corretamente o máximo de pontos, estreitando a margem (menor viés, maior variância). Os kernels — linear, polinomial, RBF/Gaussiano e sigmoide — são funções que permitem ao SVM transformar implicitamente os dados para espaços de dimensão superior onde se tornam linearmente separáveis, realizando essa projeção de forma computacionalmente eficiente através do cálculo de produtos internos, sem nunca calcular explicitamente as coordenadas no novo espaço. 

As principais vantagens dos SVMs incluem sua eficácia em espaços de alta dimensionalidade (especialmente quando o número de features supera o número de amostras), robustez contra overfitting devido à maximização da margem, flexibilidade proporcionada pelos diferentes kernels, e uso eficiente de memória já que apenas os vetores de suporte são necessários para definir a solução. Por outro lado, as desvantagens envolvem a alta complexidade computacional para grandes conjuntos de dados (o treinamento pode ser intensivo em tempo e recursos), a necessidade de seleção cuidadosa do kernel e dos hiperparâmetros (C e gamma para RBF) através de validação cruzada, sensibilidade a dados desbalanceados, e menor interpretabilidade quando comparado a modelos como árvores de decisão ou regressão logística. 

Como discutido na conexão com a aula anterior sobre ajuste de hiperparâmetros, a escolha do kernel e o ajuste de seus parâmetros são etapas cruciais que devem ser realizadas através de técnicas como grid search, random search ou otimização bayesiana, sempre utilizando validação cruzada k-fold para garantir que o modelo generalize bem para dados não vistos. Na próxima seção, introduziremos o TF-IDF (Term Frequency-Inverse Document Frequency), uma técnica de ponderação de texto essencial para aplicar SVMs em problemas de classificação textual, criando representações numéricas de documentos que capturam a importância relativa das palavras.
