# BERT

BERT (Bidirectional Encoder Representations from Transformers) é um modelo baseado na arquitetura de transformers que foi projetado especificamente para entender texto, e não para gerá‑lo token a token como um modelo de linguagem causal. Em vez de olhar apenas para a esquerda ou para a direita, o BERT usa um empilhamento de camadas de self‑attention bidirecional para construir embeddings contextuais em que cada token “enxerga” toda a frase, antes e depois dele. Esses embeddings servem como uma base poderosa para uma grande variedade de tarefas de PLN, como classificação de texto, análise de sentimentos, resposta automática a perguntas e detecção de intenção, bastando acoplar camadas simples de saída (por exemplo, um classificador) sobre as representações produzidas pelo encoder transformer.


## ***Masked Language Models*** (Modelos de Linguagem Oculta)


Até agora vimos o que são embeddings, transformers e como a atenção é um conceito importante no treinamento de Transformers como um modelo de Linguagem. Agora veremos um novo paradigma, o Codificador Transformer Bidirecional (*Bidirectional Transformer Encoder*), e sua versão mais usada, o BERT (*Bidirectional Encoder Representations from Transformers*). Esse modelo é treinado por meio de modelagem de lingua oculta, ou mascarada (*Masked Language Models*), onde, ao invés de prever a próxima palavra, nós ocultamos uma palavra no meio e pedimos ao modelo que advinhe a palavra dadas as outras palavras, antes e depois. Isso permite que o modelo veja o contexto nas duas direções (esquerda para direita e direita para esquerda).

Também veremos o que é o ***fine-tuning*** de um modelo, ou ***transfer-learning***, onde pegamos a rede de um Transformer aprendida por esses modelos pré-treinados, adicionamos uma camada para classificação na camada superior do modelo, e treinamos ele para uma nova tarefa usando um banco de dados rotulado, seja para análise de sentimento, detecção de posicionamento, etc.

## Codificadores Transformer Bidirecionais (*Bidirectional Transformer Encoder*)


```{video} https://www.youtube.com/embed/GDN649X_acE?si=mUx7D7ywLA6Isb6w
```

O foco de Codificadores Bidirecionais está em calcular representações contextualizadas dos tokens de entrada. Os codificadores bidirecionais usam "auto-atenção" (*self-attention*) para mapear a sequência dos embeddings de entrada ($X_1, ..., X_n$) às sequências de embeddings de saída do mesmo tamanho ($\mathbf{h_1, h_2, ... , h_n}$), onde os vetores de saída são contextualizados usando a informação da sequência inteira. Esses embeddings de saída são representações contextualizadas de cada token de entrada, sendo úteis para várias aplicações onde temos de classificar ou tomar decisões com base no token dentro de um contexto.


### Arquitetura

Os modelos de linguagem baseados em transformers bidirecionais diferem de duas maneiras dos transformers originais: 1) a função de atenção não é causal (isto é, somente da esquerda para direita), 2) A previsão da palavra é de uma palavra aleatoriamente selecionada "no meio" da frase, não a próxima ou a final.



```{figure} ../aula13/images/jurfig9.1.png
---
width: 100%
name: transformerbi
align: center
---
(a) O transformer causal do Capítulo 8, destacando o cálculo de atenção no token 3. O valor de atenção em cada token é calculado usando apenas as informações vistas anteriormente no contexto. (b) Fluxo de informação em um modelo de atenção bidirecional. Ao processar cada token, o modelo presta atenção a todas as entradas, tanto antes quanto depois da atual. Assim, a atenção para o token 3 pode se basear em informações de tokens seguintes.. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 198).
```

### Treinamento de Modelos de Linguagem com Transformers Bidirecionais

O modelo 'causal' do transformer era treinado prevendo a próxima palavra do texto. Com o Transformer Bidirecional, a tarefa do modelo de linguagem é a de preencher espaços em branco na frase. Por exemplo, ao invés de prever que

"A água da Praia de Copacabana é tão _________"

O modelo vai prever um item em branco, dada toda a frase:

" A _____ da Praia de Copacabana é tão ______"

Isto é, dado uma frase de entrada com um ou mais elementos faltando, a tarefa é a de preencher os elementos faltantes. Durante o treinamento, o modelo não receberá alguns tokens de entrada e deve gerar uma distribuição de probabilidades para cada um dos elementos faltantes. Alguns exemplos de distribuições de probabilidades:

1) Para a frase [A ______ da Praia de Copacabana é tão ______], no primeiro espaço o modelo pode produzir algo como:

- P(água) = 0,72  
- P(areia) = 0,15  
- P(orla) = 0,08  
- P(praia) = 0,03  
- P(outros) = 0,02  

2) Para o segundo espaço na mesma frase:

- P(limpa) = 0,40  
- P(gelada) = 0,30  
- P(poluída) = 0,20  
- P(agradável) = 0,07  
- P(outros) = 0,03  

3) Outro exemplo, com a frase [O turista ficou muito ______ com a beleza da praia]:

- P(impressionado) = 0,55  
- P(emocionado) = 0,20  
- P(feliz) = 0,15  
- P(surpreso) = 0,06  
- P(outros) = 0,04  


### Ocultando (*Masking*) as palavras 

Os Modelos de Linguagem Mascarados, ou *Masked Language Modeling*, vão usar textos não rotulados em um grande córpus, como é o caso do BERT ({cite}`Devlin2019BERT`., 2019) e do ChatGPT. No treinamento de um MLM, o modelo vai receber uma série de frases de um córpus de treinamento em que uma porcentagem dos tokens (15\% no caso do BERT) são aleatoriamente ocultados/manipulados: Dada a frase "O almoço estava delicioso", e assumindo que "delicioso" foi o token escolhido:

1) 80% das vezes o token será substituído por um token especial [MASK], e.g. "O almoço estava [MASK]"

2) 10% das vezes ele será substituído por outro token amostrado aleatoriamente: "O almoço estava [surpreso]"

3) 10% das vezes ele não sofrerá alteração: "O almoço estava delicioso".


```{figure} ../aula13/images/jurfig9.3.png
---
width: 100%
name: MLMtraining
align: center
---
Treinamento de modelo de linguagem mascarado. Neste exemplo, três dos tokens de entrada são selecionados, dois dos quais são mascarados e o terceiro é substituído por uma palavra não relacionada. As probabilidades atribuídas pelo modelo a esses três itens são usadas como função de perda de treinamento. Os outros 5 tokens não desempenham nenhum papel na perda de treinamento. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 201).
```

A {numref}`Figura {number} <MLMtraining>` mostra como funciona o treinamento de um **transformer bidirecional** no esquema de *masked language modeling* (como no BERT), usando a frase "So long and thanks for all the fish".

Na parte de baixo, cada palavra de entrada é convertida em um vetor de **token embedding** somado a um **positional embedding** (p1, p2, …, p8), formando os blocos rosados rotulados como Token + Positional Embeddings. O detalhe importante é que dois tokens foram substituídos por [mask] (long e thanks) e um foi trocado por uma palavra aleatória (the no lugar de all); mesmo assim, os embeddings correspondem às posições da sequência inteira, permitindo que o encoder veja todo o contexto, inclusive antes e depois dos espaços mascarados.

Esses vetores enriquecidos de cada posição alimentam o grande bloco central, o **Bidirectional Transformer Encoder**: aqui atuam várias camadas de *self‑attention* bidirecional, de forma que a representação de cada posição $h^L_i$ leva em conta simultaneamente todas as outras palavras da frase, tanto anteriores quanto posteriores. Ao contrário de um transformer causal, nada impede que o token na posição 4 “olhe” para o token na posição 7, por exemplo; isso é essencial para inferir corretamente as palavras escondidas.

Na parte superior, apenas três posições têm uma **LM Head with Softmax over Vocabulary** conectada: as que correspondem às palavras alvo long, thanks e the. Cada cabeça de linguagem recebe a representação final daquela posição ($h^L_2, h^L_4, h^L_7$), projeta-a para o espaço do vocabulário e aplica um softmax para gerar uma distribuição de probabilidades $y_{\text{long}}, y_{\text{thanks}}, y_{\text{the}}$. Em seguida, calcula-se a **CE Loss** (entropia cruzada) apenas nesses três pontos, como $-\log y_{\text{long}}, -\log y_{\text{thanks}}, -\log y_{\text{the}}$; os demais cinco tokens não contribuem para a perda. A média dessas perdas é usada para fazer backpropagation por todo o encoder, ajustando os pesos para que o modelo fique cada vez melhor em recuperar tokens mascarados a partir do contexto bidirecional.


### Embeddings Contextuais

Dado um modelo de linguagem pré-treinado e uma frase de entrada inédita, podemos pensar nas sequências de saída do modelo como **embeddings contextuais** para cada token de entrada. Esses Embeddings Contextuais são vetores representando algum aspecto do significado de um token dentro de contexto, e pode ser usado para qualquer tarefa que precise do significado de tokens ou palavras. Formalmente, dado os tokens de entrada $x_1, ..., x_n$, nós podemos usar o vetor de saída $h^L_i$ da última camada $L$ do modelo como uma representação do significado do token $x_i$ no contexto da frase/sequência $x_1,...,x_n$.

```{figure} ../aula13/images/jurfig9.5.png
---
width: 100%
name: embeddingcont
align: center
---
A saída de um modelo no estilo BERT é um vetor de embedding contextual $h_i^L$ para cada token de entrada $x_i$. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 205).
```

A {numref}`Figura {number} <embeddingcont>`  mostra como o encoder bidirecional de um modelo ao estilo BERT produz **embeddings contextuais** para cada posição da frase. Na parte de baixo, há os tokens de entrada [CLS So long and thanks for all], cada um passado por um bloco rosado de **Token + Positional Embeddings**: o vetor de embedding da palavra (E) é somado ao embedding de posição $i$, gerando um vetor que codifica ao mesmo tempo qual é o token e onde ele aparece na sequência. Esses vetores entram em paralelo no grande bloco roxo, o **Bidirectional Transformer Encoder**, composto por várias camadas empilhadas de self‑attention e redes feed‑forward; as “barrinhas” empilhadas em cada coluna representam essas camadas sucessivas aplicadas ao mesmo token ao longo da profundidade do modelo.

À medida que os vetores sobem pelas camadas, cada posição vai incorporando informação de todos os outros tokens da frase, para a esquerda e para a direita, graças à atenção bidirecional. No topo, obtemos um embedding final $h^L_i$ para cada posição $i$: $h^L_{\text{CLS}}$ resume a informação global da sentença inteira (é o vetor normalmente usado para tarefas de classificação de texto), enquanto $h^L_1, h^L_2, \dots, h^L_6$ são os **embeddings contextuais** de cada palavra individual, já ajustados ao contexto em que aparecem. Esses vetores podem ser reutilizados em diversas tarefas downstream, como rotular sentenças, classificar intenções ou fazer *token classification* (por exemplo, NER), mostrando como o transformer funciona como uma máquina de gerar representações ricas e específicas de contexto para cada token.


## Fine-tuning (ou Transfer Learning) para classificação de texto.

Uma vez pré-treinado como modelo de linguagem mascarada, o BERT passa a funcionar como um grande extrator de **embeddings contextuais** que podem ser adaptados a tarefas específicas por meio de *fine‑tuning*. A ideia é simples: em vez de treinar um modelo do zero, aproveitamos o encoder bidirecional já treinado em um córpus massivo e adicionamos, sobre a saída correspondente ao token [CLS], uma pequena cabeça de classificação (por exemplo, uma camada densa com softmax) que aprende a mapear o embedding global da sentença para rótulos como positivo/negativo, spam/não spam ou categorias temáticas. Durante o *fine‑tuning*, todos os pesos do BERT são ajustados levemente, em conjunto com essa camada de saída, usando um conjunto de dados rotulado da tarefa de interesse; na prática, isso significa que o mesmo modelo BERT de base pode ser reutilizado para dezenas de problemas diferentes (classificação de notícias, análise de sentimentos em avaliações, detecção de intenção em *chatbots*, entre outros), bastando trocar o cabeçalho de saída e realizar alguns poucos epochs de treinamento supervisionado em cima dos embeddings contextuais já aprendidos.


```{figure} ../aula13/images/jurfig9.9.png
---
width: 100%
name: finetuning
align: center
---
Classificação de sequência com um codificador transformer bidirecional. O vetor de saída para o token [CLS] serve como entrada para um classificador simples. Fonte: Jurafsky e Martin, 2025. ({cite}`jurafsky2024speech`., p. 210).
```

A {numref}`Figura {number} <finetuning>` mostra como fica o **fine-tuning do BERT para classificação de sentimento** usando uma frase de entrada. Na parte de baixo, cada token da sentença ([CLS], entirely, predictable, and, lacks, energy) é transformado em um embedding de token (E) somado ao embedding de posição $i$, formando os blocos rosados que alimentam o **Bidirectional Transformer Encoder**. Dentro desse encoder, as camadas de self‑attention bidirecional produzem, para cada posição, um embedding contextual; em especial, o vetor $h_{\text{CLS}}$ no topo da primeira coluna resume a informação da frase inteira e é usado como representação global do texto.

Acima desse vetor $h_{\text{CLS}}$ aparece a **sentiment classification head**, uma pequena rede (com matriz de pesos $W_c$) que projeta o embedding do [CLS] em uma distribuição de probabilidade $y$ sobre as classes de saída, por exemplo positivo vs. negativo. Durante o fine‑tuning, tanto os pesos do encoder bidirecional quanto os da cabeça de classificação são ajustados com base em exemplos rotulados (frases com seu sentimento), de modo que o modelo aprenda a associar determinados padrões nos embeddings contextuais à classe correta de sentimento para novas frases.


## Conclusão

Ao longo desta aula vimos como embeddings distribucionais, a arquitetura de Transformers e o pré‑treinamento bidirecional do BERT se encaixam em uma mesma história: primeiro representamos palavras como vetores densos, depois usamos camadas de self‑attention para torná‑los contextuais e, por fim, adaptamos esses vetores a tarefas específicas por meio de fine‑tuning. No caso de classificação de texto em português, essa receita já está embutida em modelos pré‑treinados como o BERTimbau, que seguem a mesma arquitetura do BERT original, mas foram treinados em grandes córpus em língua portuguesa; assim, no próximo passo, basta acrescentar uma cabeça de classificação sobre o token [CLS] do BERTimbau e treinar essa camada (junto com um leve ajuste dos pesos do encoder) em um conjunto rotulado em português, obtendo rapidamente classificadores de sentimentos, tópicos ou posicionamento. Veremos no tutorial da aula de hoje como aplicar o BERTimbau para a classificação de relevância dos tweets.