
# Seleção e Representação


````{margin}
```{note}
É possível que o pesquisador tenha interesse em utilizar um córpus criado por terceiros. Nesses casos, é importante analisar se esse material é adequado para responder à sua pergunta de pesquisa. Assim como em levantamentos (surveys), o córpus foi coletado e anotado com objetivos específicos em mente. Por isso, é necessário compreender suas limitações e até onde ele pode ser utilizado em seu próprio trabalho.
```
````

Como foi dito anteriormente, novas oportunidades e ferramentas de análise de texto estão disponíveis para os pesquisadores das humanidades e ciências sociais. O volume e a velocidade em que novos dados textuais e documentos são disponibilizados aumenta diariamente. Agências governamentais podem utilizar documentos e relatórios para a melhoria dos serviços públicas; Cientistas políticos podem mensurar mudanças de tópicos de debate entre os políticos, como certos tópicos flutuam ao longo do tempo nas redes sociais, etc. As possibilidades de pesquisa são inúmeras, assim como as de erros. Muitas são as dificuldades e desafios para os pesquisadores no momento de pensar em possíveis perguntas e desenhos de pesquisa, além de como operacionalizar esses dados da forma mais adequada para o seu projeto. Portanto, o pesquisador deve pensar com muito cuidado em qual sua pergunta de pesquisa, população e quantidades de interesse, o universo de documentos, e se esses documentos podem responder a sua pergunta e/ou se refletem seus interesses de pesquisa.


Na nossa disciplina, focaremos em como fazer a pesquisa utilizando métodos de aprendizado de máquina supervisionado. Para que uma aplicação desse tipo seja possível é necessário um córpus anotado. Córpus (Corpora, no plural) são conjunto de textos coletados de forma sistemática, representativo de uma língua ou variedade linguística específica, utilizado para alimentar, treinar, testar ou validar modelos e técnicas de análise automática de linguagem humana. Portanto, antes mesmo de pensar em qual técnica de aprendizado utilizar, o pesquisador precisa refletir profundamente sobre como irá construir o seu córpus e de qual universo serão retirados os documentos textuais. A qualidade, a representatividade e a precisão das anotações têm impacto direto na performance e na generalização dos modelos treinados, assim como das inferências e resultados da pesquisa.


```{admonition} 💬 Com a palavra, os autores:
:class: quote
"Os dados textuais refletem interações sociais, transações econômicas e processos políticos. Para utilizar essa riqueza de informações para formular e responder perguntas interessantes, o pesquisador deve primeiro selecionar cuidadosamente o corpus de interesse e, em seguida, representar esses documentos de forma numérica. Embora seja mais complicado devido à enorme quantidade de informações armazenadas nos textos, as decisões sobre como coletar e representar numericamente textos são semelhantes às decisões que pesquisadores tomam para representar numericamente outras variáveis de interesse nas ciências sociais. Coletar um corpus é análogo a identificar uma amostra de uma população de interesse."
({cite}`grimmer2022text`, p. 72, tradução nossa)
```

## Princípios de Seleção e Representação

O capítulo 3 do livro de Grimmer et al. aprofunda a importância de selecionar e representar textos de modo criterioso para transformá-los em dados úteis para pesquisa social. Pesquisadores precisam considerar uma série de fatores 

### 1. Construção do Corpus Guiada pela Pergunta

- O corpus deve ser construído a partir da pergunta de pesquisa.
  - Antes de coletar textos, defina claramente o que você quer saber e qual população deseja analisar.
  - Exemplo: analisar tweets pode ser ótimo para estudar engajamento político online, mas ruim para medir opinião pública geral.
- Corpus exploratório: às vezes, você começa com um corpus disponível e só depois define a pergunta. Nesse caso, é importante revisitar e refinar o corpus conforme a pesquisa avança.

### 2. Não Existe Corpus Neutro

- Toda seleção de textos envolve escolhas e valores.
  - Fatores como acesso desigual à produção de textos, políticas de preservação, censura e privacidade influenciam o que está disponível.
  - Atenção ética: uso de textos públicos nem sempre elimina preocupações com privacidade. Considere o contexto de produção e expectativas dos autores.
- Implicações práticas: um corpus pode reforçar vieses sociais, como sub-representação de certos grupos.

### 3. Não Há Uma Única Representação Correta

- A representação do texto deve ser adequada à pergunta de pesquisa.
  - Pode variar de indicadores simples (presença/ausência de palavras) a modelos complexos que capturam contexto e semântica.
  - Escolha a representação mais simples que responda sua pergunta, mas esteja aberto a ajustar conforme necessário.
- Exemplo didático: para identificar autoria, contar palavras de ligação ("e", "mas", "porém") pode ser mais útil do que analisar temas.

### 4. Validação é Essencial

- Valide suas escolhas de representação e corpus.
  - Compare resultados com dados codificados manualmente, avalie acurácia preditiva e tente replicar experimentos.
  - Se a validação falhar, revise sua abordagem.

### Exemplos Práticos

- **Discursos do Estado da União (EUA):** Permitem estudar tendências de linguagem política ao longo de séculos, mostrando como a escolha do corpus e da representação afeta as descobertas.
- **Federalist Papers:** Atribuição de autoria baseada em padrões sutis de linguagem, ilustrando a importância da representação adequada ao objetivo.

---

## Capítulo 4: Seleção de Documentos

O capítulo 4 aprofunda o processo de escolha dos textos para análise, destacando que essa etapa é fundamental para garantir inferências válidas.

### 1. População e Quantidades de Interesse

- Pergunta e população de interesse: sempre relacione sua pergunta à população que deseja estudar.
- Quantidades de interesse: defina quais métricas ou resumos você quer extrair dos textos (ex: frequência de temas, polaridade de sentimentos).

### 2. Quatro Tipos de Viés na Seleção de Corpus

| Tipo de Viés           | Descrição                                                                                       | Exemplo Didático                                   |
|------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------|
| **Viés de Recursos**   | Grupos com mais acesso a recursos produzem e preservam mais textos.                            | Arquivos históricos tendem a privilegiar elites.   |
| **Viés de Incentivo**  | Motivações estratégicas afetam o que é registrado ou omitido.                                  | Políticos podem evitar registrar discussões sensíveis. |
| **Viés de Meio**       | O formato e a tecnologia influenciam o conteúdo textual.                                       | Limite de caracteres no Twitter molda o discurso.  |
| **Viés de Recuperação**| Métodos de busca (palavras-chave, APIs) podem excluir textos relevantes ou incluir irrelevantes.| Usar só "fantasma" para buscar histórias pode perder "assombração". |

### 3. Dados Encontrados ("Found Data")

- Muitos corpora são dados encontrados, não planejados.
  - Isso impõe limitações para generalização e pode introduzir vieses difíceis de corrigir.
- Transparência: sempre explique como os textos foram selecionados e quais limitações existem.

### 4. Considerações Didáticas

- Corpus representativo: o ideal é que o corpus reflita bem a população de interesse, mas nem sempre isso é possível.
- Iteratividade: o processo de seleção pode precisar ser repetido conforme a pesquisa evolui.
- Mudanças ao longo do tempo: plataformas digitais mudam rapidamente, o que pode afetar comparações históricas.

---

## Dicas Práticas e Reflexões

- Antes de coletar textos: defina sua pergunta, população e quantidades de interesse.
- Durante a coleta: esteja atento aos vieses e limitações, documentando suas escolhas.
- Na representação: teste abordagens simples primeiro e valide sempre.
- Ao analisar: lembre-se de que textos refletem processos sociais complexos — seja crítico e transparente sobre o que seus dados realmente representam.


## Conclusão

Construir e representar um corpus de textos é um processo fundamental, cheio de decisões estratégicas e éticas. A chave está em alinhar cada etapa à pergunta de pesquisa, ser transparente sobre limitações e validar constantemente suas escolhas. Com isso, é possível transformar textos em dados realmente úteis para entender fenômenos sociais complexos.


