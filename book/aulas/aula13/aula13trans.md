# Transformers

```{video} https://www.youtube.com/embed/zxQyTK8quyY?si=YdMKOX3Ig_3kbq8A
```

Agora vamos dar um pouco de atenção aos *Transformers*, a arquitetura padrão para construir modelos *LLM* e modelos como o *BERT*. Vamos focar no uso de transformers "left-to-right" primeiro, ou transformers causais, em que temos uma sequência de tokens de entrada e fazemos previsões de tokens de saída, um por um, condicionando ao contexto anterior. O *Transformer* é uam Rede Neural com uma estrutura específica que inclui um mecanismo de "auto-atenção" (*self-attention*). A atenção pode ser pensada como uma forma de construir representações contextuais do significado de um token, prestando atenção e inegrando informações de *tokens* circundantes, ajudando o modelo a aprender como os tokens se relacioanm entre si em grandes extensões.


## Transformers


```{video} https://www.youtube.com/embed/wjZofJX0v4M?si=C1JZQgaqXvhl1XaY
```


A {numref}`Figura {number} <transformer>` ilustra a arquitetura de um ***Transformer***. Um *transformer* tem três componentes principais. No centro estão colunas de blocos de transformer ("*Stacked Transformer Blocks*" na figura). Cada um desses blocos é uma Rede Neural com múltiplas camadas, compostas de uma camada de *multi-head attention*, redes neurais *feedforward* (progressivas), e uma camada de normalização. Esses blocos vão mapear um vetor $\mathbf{X_i}$ de entrada na coluna $i$ (correspondente ao token $i$) à um vetor de saída $\mathbf{h_i}$. Os n blocos de atenção mapeiam a janela inteira de vetores de entrada ($X_1, ..., X_n$) à uma janela de vetores de saída ($h_1,..., h_n$) de mesmo tamanho. 

```{figure} ../aula13/images/jurfig8.1.png
---
width: 100%
name: transformer
align: center
---
A arquitetura de um transformer (da esquerda para a direita), mostrando como cada token de entrada é codificado, passado por um conjunto de blocos transformer empilhados e, em seguida, por uma cabeça de modelo de linguagem que prevê o próximo token.. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 171).
```

A coluna de bolcos de atenção é precedida de um componente de codificação da entrada, que processa cada token de entrada em um vetor de representação. E cada coluna é seguida de uma cabeça de modelo de linguagem, que pega a saída do embedding do bloco final do transformer, passa isso por uma matriz de "*unembedding*" $\mathbf{U}$ (o inverso dos embeddings de entrada).Uma função softmax então converte esses valores em probabilidades, prevendo o token mais provável a ser gerado em seguida, como na geração autoregressiva de texto.


## Atenção 


```{video} https://www.youtube.com/embed/eMlx5fFNoYc?si=qrYEYFtiG2US51sL
```

Na seção anterior (*Embeddings*), vimos que o *Word2Vec* (e outras representações estáticas) tem representações do significado de uma palavra iguais independente do contexto. No entanto, sabemos que o contexto importa, e o significado das palavras pode variar bastante a depender deste contexto. Os *Transformers* nos permiter capturar o contexto e construir vetores de representação contextuais. Em um *transformer* conseguimos construir representações dos significados de vetores de entrada mais ricos e com contextualização.

A **atenção** (*attention*) é o mecanismo pelo qual o transformer vai ponderar e combinar as representações de outros tokens apropriados no contexto da camada $k$ para construir a representação dos tokens na camada $k+1$.



```{figure} ../aula13/images/jurfig8.2.png
---
width: 100%
name: attention
align: center
---
A distribuição de pesos de autoatenção α que faz parte do cálculo da representação da palavra "it" na camada k+1. Ao computar a representação de "it", o modelo dá atenção diferente a várias palavras da camada k, com tons mais escuros indicando valores de autoatenção mais altos. Note que o transformer presta alta atenção às colunas correspondentes aos tokens "chicken" e "road", um resultado sensato, pois no ponto em que "it" ocorre, ele poderia plausivelmente se referir ao "chicken" ou ao "road", e portanto desejamos que a representação de "it" incorpore as representações dessas palavras anteriores. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 173).
```

A {numref}`Figura {number} <attention>` mostra um esquema simplificado do mecanismo de atenção de um transformer. A situação representada é a do cálculo de uma representação contextual para o token k+1, usando a representação do token em k e os anteriores. A figura usa cor para representar a distribuição de atenção sobre as palavras contextuais. Os tokens de "chicken" e "road" têm um alto peso de atenção, significando que, ao calcular a representação da palavra "it", daremos maior peso a estas duas palavras.



```{figure} ../aula13/images/jurfig8.3.png
---
width: 100%
name: attentioncausal
align: center
---
Fluxo de informação na autoatenção causal. Ao processar cada entrada $x_i$, o modelo presta atenção a todas as entradas até $x_n$, incluindo $x_i$. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 174).
```

A {numref}`Figura {number} <attentioncausal>` mostra uma camada de self-attention em um transformer, em que cada token de entrada $x_i$ é transformado em uma nova representação $a_i$. Dentro da faixa roxa (“Self-Attention Layer”), cada bloco de atenção recebe um token de entrada, mas olha simultaneamente para todos os tokens da sequência, o que é indicado pelas várias setas que saem de cada $x_j$ e chegam ao bloco de atenção responsável por um $a_i$. Assim, a representação de saída $a_i$ é uma combinação ponderada de todos os tokens de entrada, permitindo que o modelo leve em conta o contexto completo da frase ao atualizar cada posição.

```{figure} ../aula13/images/jurfig8.16.png
---
width: 100%
name: transfmodling
align: center
---
Treinando um transformer como um modelo de linguagem. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 174).
```


A {numref}`Figura {number} <transfmodling>` mostra, passo a passo, como um transformer é treinado como modelo de linguagem autorregressivo: ele recebe uma sequência de tokens de entrada e aprende a prever, em cada posição, qual é o próximo token da sequência. Cada coluna da esquerda para a direita corresponde a uma posição temporal $t$ (por exemplo, os tokens So, long, and, thanks, for), e em cada uma delas o modelo produz uma distribuição de probabilidade sobre todo o vocabulário, usada para calcular a perda de linguagem naquela posição.[^1][^2]

Na parte de baixo, os **Input tokens** entram no bloco de **Input Encoding**: cada palavra é mapeada para um vetor de embedding $E$ e somada a um embedding de posição (1, 2, 3, 4, 5), gerando os vetores $x_1, x_2, \dots, x_5$ que carregam ao mesmo tempo identidade do token e sua posição na frase. Esses vetores $x_t$ são então alimentados, em paralelo, na pilha de **Stacked Transformer Blocks** (os blocos roxos), que implementam a autoatenção causal: para calcular a representação em uma coluna, o modelo “olha” apenas para os tokens até aquela posição (setas diagonais ligando entradas anteriores a blocos posteriores), acumulando gradualmente o contexto.[^2][^3]

No topo de cada coluna, aparece a **Language Modeling Head**: é uma camada linear com matriz de “unembedding” $U$ que transforma a representação final de cada posição em logits sobre o vocabulário, seguida de um softmax para obter probabilidades $y_{\text{long}}, y_{\text{and}}, \dots$. Em seguida, cada coluna calcula uma perda de entropia cruzada $-\log y_{\text{token correto}}$, comparando a probabilidade atribuída ao token correto (por exemplo, prever long dado So, prever and dado So long, etc.). A expressão à direita indica que a **loss total** do exemplo é a média dessas perdas ao longo de todas as posições $t$, e é esse valor que é minimizado via backpropagation para ajustar todos os pesos do encoder de entrada, dos blocos transformer e da cabeça de linguagem, treinando o transformer como modelo de linguagem que prevê o próximo token.

## Conclusão

Nesta seção vimos como a arquitetura de Transformers define o “motor” básico dos modelos modernos de linguagem: embeddings de entrada, camadas empilhadas de self‑attention e uma cabeça de saída que transforma representações contextuais em previsões de tokens. Ao treinar esse motor como modelo de linguagem, aprendemos vetores que capturam ricas relações semânticas entre tokens em contexto, e é exatamente esse mecanismo que o BERT aproveita ao usar apenas a parte de encoder do transformer, de forma bidirecional. A partir dessas representações, fica simples “plugar” uma camada de classificação em cima do [CLS] (ou da média dos embeddings) para tarefas supervisionadas como análise de sentimento, detecção de spam ou classificação temática de documentos, mostrando como o transformer funciona como bloco fundamental tanto para geração quanto para compreensão e classificação de textos. É isso que veremos nos próximos blocos desta aula.