# Modelos de Linguagem "Grandes" (LLMs)

## Conversando com Robos

Nas aulas anteriores, exploramos os fundamentos que nos trouxeram até aqui: embeddings estáticos como Word2Vec, que representam palavras como vetores densos; a arquitetura de Transformers, com seu mecanismo de self-attention que permite capturar contexto; e o BERT, que usa encoders bidirecionais para criar representações contextuais poderosas por meio de masked language modeling. Agora, daremos o próximo passo nessa jornada, estudando os **Large Language Models (LLMs)** — modelos generativos de grande escala que revolucionaram o campo de Processamento de Linguagem Natural.


O primeiro chatbot criado foi a ELIZA, em 1966, e simulava um psicólogo Rogeriano: Isto é, invertia as entradas dos usuários na forma de pergunta.  Nesse modelo, o sistema basicamente devolvia as falas do usuário na forma de perguntas, invertendo pronomes e reformulando a frase para encorajar a continuação da conversa, como em: “Estou muito preocupado com o meu trabalho” → “Por que você está tão preocupado com o seu trabalho?”. 

```{figure} ../aula14/images/eliza.png
---
width: 100%
name: eliza
align: center
---
Exemplo de conversação com a ELIZA. Ainda(!) disponível para conversa em: https://web.njit.edu/~ronkowit/eliza.html
```

Mesmo com essa limitação, muitos usuários da ELIZA sentiram que ela apresentava bastante capacidade conversacional, e alguns apresentaram apego emocional ao chatbot, entrando em conversas bem pessoais e pedindo para seu criador dar "privacidade" para a conversa (Jurafsky, 2025). Outro exemplo recente de chatbot que ganhou certa notoriedade foi o SimSimi, que aprendia respostas com os usuários, o que acabou gerando problemas e controvérsia.

```{figure} ../aula14/images/simsimi.jpg
---
width: 100%
name: simsimi
align: center
---
"App SimSimi preocupa pais por conteúdo impróprio para crianças; saiba riscos". Disponível em [Estadão](https://www.estadao.com.br/brasil/app-simsimi-e-criticado-por-conteudo-inadequado-para-criancas-saiba-riscos/?srsltid=AfmBOoo9hb95AUeJA8xHRRzz0p2Tr_r4P7A8iLH_0NYpVYxdM8QwXVNm)
```

Muitas décadas se passaram desde a ELIZA, e chegamos no chatbot mais recente e com ampla adoção e discussão: o ChatGPT e correlatos. Esses chatbots recentes são baseados em LLMs, *Large Language Models*, ou "Modelos de Linguagem Grandes" em português. A principal ideia por trás desse tipo de modelo é a ideia do pré-treinamento com grandes volumes de texto. Uma *LLM* é uma Rede Neural desenhada para entender, gerar, e responder textos. O "Grande" de seu nome se refere tanto ao volume de texto em que são treinados quanto ao número de parâmetros estimados:

1. **Tamanho em parâmetros**: Possuem dezenas ou centenas de bilhões de parâmetros ajustáveis
2. **Tamanho dos dados de treinamento**: São treinados em enormes quantidades de texto, frequentemente incluindo grandes porções da internet pública

**TABELA: Comparação de LLMs (GPT, LLaMA, Gemma, etc.), Xiao & Zhu, 2025, p. 41**

| LLM | # Parâmetros | Profundidade L | Largura d | # Heads (Q/KV) |
|-----|--------------|----------------|-----------|----------------|
| GPT-2 | 1.5B | 48 | 1,600 | 25/25 |
| GPT-3 | 175B | 96 | 12,288 | 96/96 |
| LLaMA2-7B | 7B | 32 | 4,096 | 32/32 |
| LLaMA2-70B | 70B | 80 | 8,192 | 64/64 |
| LLaMA3-405B | 405B | 126 | 16,384 | 128/8 |

Fonte: {cite}`xiao2025foundations`, p. 41.


*LLMs* usam o transformer como arquitetura base e, visto que são capazes de gerar texto, são parte da nova área da IA conhecida como "IA Generativa".


```{figure} ../aula14/images/raschkafig1.1.png
---
width: 100%
name: genai
align: center
---
Divisão das áreas da Inteligência Artificial. Fonte: Raschka (2024, {cite}`raschka2024build`. ,p.3)
```

## LLMs

```{video} https://www.youtube.com/embed/LPZh9BOjkQs?si=Ok5nkoos0_t1o5AI
```

Os *LLMs*, como diz o nome, são modelos grandes de linguagem e, portanto, aprendem vocabulário, contexto e palavras novas por meio da tarefa principal de um modelo de linguagem: Prever a próxima palavra. O que, de fato, os *LLMs* podem aprender com previsão de palavras?


```{figure} ../aula14/images/jurfig7.1.png
---
width: 100%
name: word2vecemb
align: center
---
Modelo de Linguagem. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`. ,p.147)
```

A partir desse treinamento em grandes volumes de textos, o modelo aprende que rosas, dálias e violetas são flores que ocorrem em contextos semelhantes. Da mesma forma, compreende que "grande" e "enorme" ocupam uma extremidade de uma escala, enquanto "pequeno" e "minúsculo" posicionam-se na extremidade oposta. Contudo, o modelo também absorve vieses e preconceitos presentes nos textos de treinamento. Pode, por exemplo, associar a categoria "médico" ao gênero masculino, preterindo o uso de "médica" ou presumindo que o usuário se refere a um homem. Este é apenas um exemplo entre diversos tipos de associações espúrias e enviesadas que podem ser internalizadas por LLMs e outros modelos submetidos a pré-treinamento em textos produzidos por humanos, os quais carregam consigo esses vieses e preconceitos. A situação torna-se ainda mais crítica quando a IA é aplicada em contextos particularmente suscetíveis a esse tipo de viés, como dados jurídicos, policiais e de outras áreas sensíveis. Isso levanta questões éticas importantes, especialmente dado que modelos generativos podem ter usos escusos (*deepfakes*, golpes etc.) e serem aplicados de maneira inconsequente em aplicações do governo ou em áreas sensíveis (reconhecimento facial de criminosos, detecção de doenças, etc.). Na aula de hoje, falaremos especialmente das LLMs e o uso de IA generativa para texto, mas essas preocupações se mantêm.

A intuição principal por trás das LLMs, e modelos de linguagem de forma geral, é a de que o mesmo modelo que pode prever a próxima palavra, criando uma distribuição de palavras prováveis, pode ser também utilizado para gerar texto amostrando palavras de dentro dessas distribuições de probabilidades estimadas.

```{figure} ../aula14/images/jurfig7.2.png
---
width: 100%
name: word2vecemb
align: center
---
Probabilidades em um Modelo de Linguagem. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`. ,p.147)
```

A ideia de usar modelos computacionais para gerar texto, assim como códigos, imagens, falas etc., se consolidou em uma nova área dentro da IA, a IA generativa (mais conhecida como *GenAI*). Os LLMs, como GPT-3, GPT-4, LLaMA e outros, são modelos de linguagem autorregressivos baseados em Transformers que foram treinados em quantidades massivas de texto (trilhões de tokens) para prever a próxima palavra em uma sequência. Diferentemente do BERT, que é um modelo de *compreensão* de texto (encoder-only), os LLMs são modelos *generativos* (decoder-only) projetados para produzir texto de forma fluente e coerente, token por token.


## Três Arquiteturas de Modelos de Linguagem

As três principais arquiteturas de *LMs* são as de *Encoders*, *Decoders* e *Enconder-Decoders*.

```{figure} ../aula14/images/jurfig7.3.png
---
width: 100%
name: arquiteturaslms
align: center
---
Arquiteturas de Modelo de Linguagem. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`. ,p.148)
```

O ***Decoder*** é a arquitetura causal que vimos na aula anterior. Isto é, ela iterativamente percorre a frase da esquerda para a direita, calculando as probabilidades. O *decoder* é a arquitetura usada nos principais *LLMs*, como o *GPT*, o *Claude*, *LLama* etc. *Decoders* são a base do modelo generativo.

Já os ***Encoders*** pegam como entrada uma sequência de tokens e têm como saída uma representação vetorial para cada tokens. Essa arquitetura nós vimos no treinamento de MLMs e do BERT. *Encoders* geralmente são *MLM* (*Masked Language Models*) e são a base de modelos como o BERT, RoBERTA, etc. São modelos que não são usados de forma generativa, e mais usados para classificação e outras tarefas supervisionadas com texto.

Os **Encoder-decoders** são arquiteturas mais utilizadas para reconhecimento de fala e tradução. Nessa configuração, o *encoder* processa toda a sequência de entrada (por exemplo, uma frase em inglês ou um sinal de áudio) e gera representações contextualizadas. O *decoder* então recebe essas representações e gera a sequência de saída token por token (por exemplo, a tradução em português ou a transcrição do áudio). A arquitetura original do Transformer, apresentada no artigo "Attention is All You Need", era justamente um encoder-decoder. Modelos como T5, BART e os sistemas de tradução neural modernos utilizam essa arquitetura, pois ela permite que o modelo "compreenda" completamente a entrada antes de começar a gerar a saída, sendo ideal para tarefas de sequência-para-sequência.


### Diferenças principais entre BERT e GPT:

| Característica | BERT (Encoder) | GPT (Decoder) |
|----------------|----------------|---------------|
| **Tipo de atenção** | Bidirecional | Causal (unidirecional) |
| **Objetivo de treino** | Masked Language Modeling | Next-Token Prediction |
| **Uso principal** | Compreensão/Classificação | Geração de texto |
| **Token especial** | [CLS], [MASK] | Tokens de contexto |



## Geração Condicional de Texto

A geração condicional de texto é a ideia de que o modelo sempre escreve “em resposta a alguma coisa”: um contexto, uma instrução, um documento ou até uma parte anterior do próprio texto. Nesse enquadramento, praticamente qualquer tarefa de linguagem pode ser vista como “gerar texto condicionado a uma entrada”, e essa é a intuição central por trás de modelos grandes de linguagem.

Mais formalmente, fala-se em gerar uma sequência de tokens condicionada a outra sequência: dado um texto de entrada (o *prompt*), o modelo produz tokens um a um, de forma autoregressiva, de modo que cada novo token é amostrado de uma distribuição de probabilidade que depende tanto da entrada quanto de tudo o que já foi gerado até aquele ponto. Em notação probabilística, o modelo aprende algo do tipo $p(y_1, \dots, y_T \mid x)$, onde $x$ é o texto de entrada (condição) e $y_1, \dots, y_T$ são os tokens gerados.



```{figure} ../aula14/images/jurfig7.4.png
---
width: 100%
name: geracaocond
align: center
---
Geração Condicional de texto. Fonte: Jurafsky e Martin (2025, {cite}`jurafsky2024speech`. ,p.150)
```

Aqui estão alguns exemplos de geração condicional de texto:

- **Resposta a perguntas**: o usuário escreve "Explique o que é overfitting em poucas linhas", e o modelo gera um parágrafo explicativo condicionado a esse pedido; a tarefa inteira é “gerar texto (resposta) condicionado ao enunciado da pergunta”.
- **Resumo de documentos**: o modelo recebe um artigo longo como entrada e deve produzir um resumo; aqui, a sequência de saída (resumo) é condicionada ao documento de entrada.
- **Tradução automática**: na tradução, o texto em língua de origem (por exemplo, inglês) é a condição, e o texto em língua de destino (por exemplo, português) é a sequência gerada; essa é uma forma clássica de geração condicional em tarefas de sequência-para-sequência.
- **Completar código ou texto**: quando se escreve o começo de uma função ou de um parágrafo e o modelo continua, o prefixo é o contexto condicional, e a continuação é a sequência gerada.


Na prática, a geração condicional se manifesta por meio de *prompts*, que funcionam como a “condição” que guia o comportamento do modelo. Um mesmo modelo pode fazer tarefas muito diferentes alterando apenas o texto de entrada: um *prompt* formulado como instrução (“Resuma o texto a seguir em três pontos principais:”) induz uma saída estruturada em tópicos, enquanto um *prompt* narrativo (“Escreva a continuação desta história:”) induz uma continuação ficcional.

Além disso, o controle da geração (por exemplo, com *temperature*, *top-k*, *top-p*) atua sobre a forma como a distribuição condicional de probabilidades é amostrada: mesmo condicionados ao mesmo *prompt*, diferentes configurações podem produzir saídas mais conservadoras ou mais criativas, mas sempre dentro da distribuição condicionada aprendida pelo modelo.


## Prompting

A ideia de geração condicional já é bastante poderosa, mas se torna ainda mais útil quando o modelo é treinado explicitamente para responder perguntas e seguir instruções em linguagem natural. Nesse cenário, em vez de apenas completar texto, o modelo passa a “entender” melhor comandos como "explique", "resuma", "traduza" ou "liste", e a produzir saídas alinhadas a essas instruções.

```{figure} ../aula14/images/jurfig7.5.png
---
width: 100%
name: prompting
align: center
---
Geração condicional de texto para responder questões e seguir instruções. Adaptado de Jurafsky e Martin (2025, {cite}`jurafsky2024speech`, p. 151).
```

Esse tipo de treinamento adicional é conhecido como *instruction-tuning* e consiste em continuar o treinamento da LLM com um conjunto de pares instrução/pergunta–resposta, cobrindo vários tipos de tarefas e formatos de pedido. Como resultado, o modelo se torna mais capaz de seguir instruções detalhadas, manter diálogos coerentes e adaptar o estilo de resposta ao que o usuário solicita.

Chama-se ***prompt*** qualquer entrada de texto fornecida pelo usuário para indicar ao modelo o que fazer, como "explique o texto abaixo em linguagem simples" ou "gere cinco títulos possíveis para este parágrafo". O processo de formular *prompts* mais claros, específicos e eficazes para obter o comportamento desejado do modelo é conhecido como *engenharia de prompt*, e hoje é uma habilidade central para explorar ao máximo o potencial de LLMs em aplicações práticas.


## Treinando LLMs

LLMs são normalmente treinados em três grandes fases, que vão de aprender padrões gerais de linguagem até se tornarem modelos úteis, seguros e voltados para seguir instruções humanas. Em cada etapa, muda tanto o tipo de dado usado quanto o objetivo de treino do modelo.

1. **Pré-treinamento**
No primeiro estágio, o modelo é treinado para prever o próximo token (palavra ou subpalavra) em um corpus massivo de textos, usando uma função de perda como *cross-entropy* para ajustar seus parâmetros. O resultado é um modelo muito competente em modelar a distribuição da linguagem, isto é, em prever sequências plausíveis e gerar texto fluente em diversos domínios.
2. ***Instruction-tuning***
Em seguida, o modelo passa por um refinamento em que aprende a seguir instruções explícitas, responder perguntas, resumir textos, gerar código e executar outras tarefas guiadas por comandos em linguagem natural. Para isso, utiliza-se um corpus especial contendo pares instrução–resposta considerados adequados, o que faz o modelo se comportar mais como um “assistente” que entende pedidos do usuário.
3. ***Alignment***
Por fim, o modelo é ajustado para ficar mais alinhado a valores e critérios de utilidade e segurança, reduzindo comportamentos danosos ou indesejáveis. Nessa fase, o treino favorece respostas vistas como aceitáveis (por exemplo, úteis, honestas e menos ofensivas) e desestimula respostas rejeitáveis, muitas vezes usando feedback humano ou sinais de preferência para orientar o comportamento do modelo.





## Conclusão

Ao longo desta seção vimos que embeddings são o elo entre textos em linguagem natural e modelos de Deep Learning: eles convertem palavras em vetores densos que capturam semelhança semântica e, no caso de modelos como o BERT, também o contexto em que cada palavra aparece. Os métodos distribucionais clássicos, como Word2Vec com skip‑gram e amostragem negativa, já nos permitem sair de vetores esparsos de contagem e aprender representações contínuas úteis para classificadores e outros modelos supervisionados. Mas é com os embeddings contextuais baseados em transformers que essa ideia atinge todo o seu potencial: o BERT usa a pilha de self‑attention de um encoder transformer para produzir, a cada camada, vetores que dependem de toda a sentença, servindo como entradas riquíssimas para tarefas de classificação de texto, análise de sentimentos, question answering e muitas outras aplicações modernas de PLN.





