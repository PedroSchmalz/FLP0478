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

A ideia de usar modelos computacionais para gerar texto, assim como códigos, imagens, falas etc., se consolidou em uma nova área dentro da IA, a IA generativa (mais conhecida como *GenAI*). Os LLMs, como GPT-3, GPT-4, LLaMA e outros, são modelos de linguagem autorregressivos baseados em Transformers que foram treinados em quantidades massivas de texto (trilhões de tokens) para prever a próxima palavra em uma sequência [file:14][file:15][file:16]. Diferentemente do BERT, que é um modelo de *compreensão* de texto (encoder-only), os LLMs são modelos *generativos* (decoder-only) projetados para produzir texto de forma fluente e coerente, token por token.


## Três Arquiteturas de Modelos de Linguagem

As três principais ar


## Conclusão

Ao longo desta seção vimos que embeddings são o elo entre textos em linguagem natural e modelos de Deep Learning: eles convertem palavras em vetores densos que capturam semelhança semântica e, no caso de modelos como o BERT, também o contexto em que cada palavra aparece. Os métodos distribucionais clássicos, como Word2Vec com skip‑gram e amostragem negativa, já nos permitem sair de vetores esparsos de contagem e aprender representações contínuas úteis para classificadores e outros modelos supervisionados. Mas é com os embeddings contextuais baseados em transformers que essa ideia atinge todo o seu potencial: o BERT usa a pilha de self‑attention de um encoder transformer para produzir, a cada camada, vetores que dependem de toda a sentença, servindo como entradas riquíssimas para tarefas de classificação de texto, análise de sentimentos, question answering e muitas outras aplicações modernas de PLN.





