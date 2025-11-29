# Embeddings

Na última aula vimos os fundamentos do Deep Learning, com introdução aos conceitos principais de redes neurais artificiais, suas unidades básicas, funções de ativação como sigmoid e ReLU, e arquiteturas como redes de camada única, multicamadas, CNNs para imagens e RNNs para dados sequenciais, destacando a transição da engenharia manual de features para extração automática de representações. Exploramos o processo de treinamento via descida de gradiente, backpropagation, SGD com minibatches, epochs e hiperparâmetros como taxa de aprendizado, além de técnicas de regularização (L1/L2, decaimento de pesos) e dropout para evitar overfitting e mínimos locais.


Na aula de hoje íremos estudar uma nova forma de representação textual que são os embeddings, vetores densos que podem ser estáticos (Word2Vec) ou dinâmicos (BERT), e no caso deste último permitem que o modelo leve em conta o contexto dos tokens/palavras, alterando sua representação conforme o contexto. Um primeiro ponto é que os embeddings estáticos permitem dar o primeiro passo na direção do contexto das palavras. Lembre-se de aulas anteriores que representações (que também eram embeddings) baseadas na contagem de palavras (*Bag-of-words* e *TF-IDF*) não levavam em conta o contexto das palavras, permitindo somente a conjunção das palavras em bigramas, trigramas, etc. 


## Contexto


O papel do contexto é muito importante para a significação das palavras e frases. Palavras que ocorrem em contextos similares tendem a ter significados similares. Essa ligação entre similaridade na forma em que palavras são distribuídas e similaridade em seu significado é conhecida como a hipótese distribucional (que ancora a semântica distribucional). Esta hipótese foi levantada nos anos 1950 por linguistas que notaram que palavras que são sinônimos tendem a ocorrer no mesmo ambiente, com a "quantidade" da diferença entre elas sendo mais ou menos a diferença entre seus ambientes. 


## Embeddings

```{video} https://www.youtube.com/embed/viZrOnJclY0?si=RNOT-8OCvexRDeuA
```

Na aula de hoje conheceremos os ***embeddings***, ou **Representações vetoriais dos significados** das palavras que são aprendidas diretamente das distribuições de palavras presentes nos textos de treinamento. *Embeddings* são o primeiro exemplo do curso de aprendizado de representação, aprendendo automaticamente representações úteis do texto de entrada. A ideia por trás das representações vetoriais é a de representar uma palavra como um ponto em um espaço semântico multidimensional que é derivado da distribuição de palavras vizinhas. Vetores para a representação de palavras são chamados de *embeddings*, ou **incorporações** (em uma tradução bem vulgar). O termo deriva historicamente da ideia matemática de mapear de um espaço, ou estrutura, para outro.



```{figure} ../aula13/images/jurfig5.1.png
---
width: 100%
name: word2vecemb
align: center
---
Uma visualização bidimensional (t-SNE) dos embeddings word2vec de 200 dimensões para algumas palavras próximas à palavra "sweet", mostrando que palavras com significados semelhantes estão próximas no espaço. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 99).
```

A {numref}`figura {number} <word2vecemb>` mostra a visualização dos *embeddings* aprendidos pelo algoritmo *word2vec*, mostrando a posição de palavras selecionadas, projetadas de um espaço de 200 dimensões em um espaço bidimensional. Perceba que vizinhos próximos de "*sweet*" estão relacionados entre si, como "*honey*", "*juice*" e assim por diante. Essa ideia está por trás do poder dos modelos de linguagem e de PLN atuais: Ao representar palavras como *embeddings*, um classificador pode atribuir sentimento ao ver palavras com significados similares.


## Word2Vec

Nas aulas anteriores, vimos como representar uma palavra como um vetor esparso e com o comprimento igual ao n´pumero de palavras no vocabulários (*Bag-of-words*,*TF-IDF*). Agora íremos estudar como funcionam os embeddings, vetores curtos e densos.


```{figure} ../aula13/images/brplnfig.10.1.png
---
width: 100%
name: modelosdistri
align: center
---
Ilustração dos Modelos Semânticos Distribucionais. Fonte: Caseli e Nunes. ({cite}`CaseliNunes2024`., p.194))
```

A {numref}`Figura {number} <modelosdistri>` ilustra a principal diferença entre o paradigma de representação vetorial que seguimos durante o curso (vetores esparsos), e o novos vetores, ou *embeddings*, que utilizaremos agora para o treinamento de modelos de aprendizado profundo. Focaremos principalmente no Word2Vec e no BERT, e veremos um pouco do GPT na próxima aula. 

Nessa seção veremos como calcular os embeddings por meio do método skip-gram com amostragem negativa (SGNS, ou *Skip-gram with negative sampling*). Este método é um de dois algoritmos usados no pacote *Word2Vec*. Os métodos desse pacote são rápidos, fáceis de treinar, e fáceis de encontrar já com código e embeddings disponíveis. Os *embeddings* gerados pelo *Word2Vec* são estáticos. Isto é, o método aprende uma única representação fixa para cada palavra no vocabulário. Depois veremos que as representações do BERT são dinâmicas e contextuais, e o vetor para cada palavra é diferente em diferentes contextos.

A intuição por trás do *Word2Vec* é a de que, ao invés de contar a frequência de cada palavra $X$ perto da palavra $Y$, treinaremos um modelo numa tarefa de classificação binária: Quão provável é que a palavra $X$ apareça próxima da palavra $Y$? Disso pegaremos os pesos estimados pelo classificador como os *embeddings* das palavras/tokens. A intuição por trás do skip-gram é a seguinte:

1. Tratar a palavra alvo e a palavra vizinha como exemplos positivos;
2. Amsotrar aleatoriamente outras palavras do vocabulário para pegar amostras negativas;
3. Usar Regressão Logística para treinar um classificador que distinga os dois casos;
4. Usar os pesos estimados como *embeddings*.

Isso é o que está representado na {numref}`Figura {number} <skipgram>`

```{figure} ../aula13/images/brplnfig.10.5.png
---
width: 100%
name: skipgram
align: center
---
Arquitetura Skip-gram. Prediz o contexto com base na palavra-alvo.
. Fonte: Caseli e Nunes. ({cite}`CaseliNunes2024`., p.208))
```








