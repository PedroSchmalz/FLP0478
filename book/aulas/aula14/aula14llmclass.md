
## Classificação de Texto com LLMs (Decoders)

Modelos *decoder-only* (como GPT, LLaMA e afins) podem ser usados como classificadores de texto tratando a classificação como um caso especial de geração condicional: dado um *prompt* que descreve a tarefa e apresenta o texto de entrada, o modelo gera como saída o rótulo da classe. Esses modelos foram treinados para prever o próximo token condicionado a um contexto, e o *instruction-tuning* faz com que respondam bem a instruções do tipo “classifique”, “rotule”, “diga se…”, permitindo que façam classificação mesmo sem uma cabeça de *softmax* específica por classe.

Do ponto de vista probabilístico, o modelo aprende uma distribuição $p(y_1, \dots, y_T \mid x)$, em que $x$ é o *prompt* (instrução + texto) e $y_1, \dots, y_T$ são os tokens da resposta. Ao restringir a resposta esperada a rótulos curtos (“positivo”, “negativo”, “spam”, “urgente”), a tarefa de classificação vira apenas gerar poucos tokens específicos que representam as classes. A qualidade dessa classificação depende em grande parte de como formulamos o *prompt* e, em contextos supervisionados, de como acoplamos (ou não) cabeças de classificação treináveis sobre as representações internas do modelo.


## Classificação zero-shot com decoders

Na classificação *zero-shot*, não fornecemos exemplos rotulados ao modelo; apenas descrevemos a tarefa e as classes em linguagem natural e pedimos um rótulo. Essa abordagem explora diretamente o *instruction-tuning*: o modelo generaliza a partir de instruções genéricas (“classifique”, “escolha uma das opções abaixo”) para a tarefa específica.

### Exemplo 1: Análise de sentimento geral

```text
Você é um sistema de análise de sentimento.
Classifique o sentimento do texto a seguir como "positivo", "negativo" ou "neutro".
Responda apenas com uma dessas três palavras, sem explicações adicionais.

Texto: "O atendimento foi péssimo e a comida estava fria."

Sentimento:
```

Aqui, o modelo precisa entender o conceito de sentimento e de termos associados a insatisfação (“péssimo”, “fria” em contexto de comida) para escolher “negativo”.

### Exemplo 2: Classificação de tema de notícia

```text
Você é um classificador de notícias.
Dada a notícia abaixo, escolha uma das categorias a seguir:
- política
- economia
- esportes
- tecnologia
- cultura

Responda apenas com o nome de uma categoria.

Notícia: "O governo anunciou hoje um pacote de medidas para estimular a indústria e reduzir impostos sobre exportações."

Categoria:
```

Mesmo sem exemplos anteriores, o modelo costuma responder “economia” ou “política”, dependendo de como foi treinado e de como interpreta o foco da notícia.

### Exemplo 3: Classificação de urgência

```text
Você é responsável por priorizar chamados de suporte.
Classifique o nível de urgência da descrição abaixo em uma destas categorias:
- baixa
- média
- alta
- crítica

Responda apenas com uma palavra.

Descrição: "O sistema de pagamentos está fora do ar para todos os clientes desde as 8h. Não conseguimos processar nenhuma venda."

Urgência:
```

O modelo precisa relacionar “fora do ar para todos os clientes” a um impacto alto no negócio e, idealmente, responder “crítica” ou “alta”.

### Profundidade: o que está acontecendo?

Em *zero-shot*, o modelo não está “aprendendo” a tarefa naquele momento; ele está reciclando padrões que viu em seu treinamento de instruções. Isso funciona bem quando:

- As classes são conceitualmente claras (positivo/negativo, spam/não spam, etc.).
- A tarefa se parece com instruções que o modelo já viu (“classifique X como Y ou Z”).
- O texto de entrada não difere muito do domínio de treino (sem jargões altamente específicos).

Por outro lado, *zero-shot* tende a ter dificuldades quando as classes são muito sutis (“ironia leve” vs “crítica severa”), quando o domínio é especializado (jurídico, médico) ou quando é necessário seguir critérios de classificação muito idiossincráticos (por exemplo, padrões internos de uma empresa).


## Classificação few-shot com exemplos no prompt

Na classificação *few-shot*, colocamos alguns exemplos rotulados dentro do próprio *prompt*, antes do exemplo que queremos classificar. O modelo observa o padrão “Texto: … → Classe: …” e tenta continuar esse padrão no novo caso. Isso pode ser visto como uma forma de *learning from context*: não mudamos os pesos do modelo, mas damos um “minicurso” dentro do *prompt*.

### Exemplo 1: Sentimento com nuances

```text
Você é um sistema de análise de sentimento para avaliações de restaurantes.
Classifique cada texto como "positivo", "negativo" ou "neutro".
Veja os exemplos e siga o mesmo padrão.

Exemplos:

Texto: "A comida estava maravilhosa e o atendimento foi impecável."
Sentimento: positivo

Texto: "A comida é aceitável, nada de especial, mas pelo preço está ok."
Sentimento: neutro

Texto: "Esperei mais de uma hora e meu pedido veio errado."
Sentimento: negativo

Agora, classifique o texto abaixo:

Texto: "A sobremesa é deliciosa, mas o prato principal veio frio e sem tempero."
Sentimento:
```

O modelo precisa perceber que há pontos positivos (sobremesa “deliciosa”) e negativos (prato principal “frio e sem tempero”). Dependendo da nuance que se deseja, pode ser treinado (via exemplos) a considerar o todo como “negativo” ou “neutro”. Se os exemplos de treinamento mostraram que problemas com o prato principal pesam mais, o modelo tende a escolher “negativo”.

### Exemplo 2: Intenção em chatbot de suporte

```text
Você é um assistente que identifica a intenção dos usuários.
Classifique a mensagem em uma das categorias:
- dúvida
- reclamação
- elogio
- cancelamento

Exemplos:

Mensagem: "Gostei muito do atendimento, vou recomendar para meus amigos."
Intenção: elogio

Mensagem: "Vocês atrasaram a entrega de novo, isso é um absurdo."
Intenção: reclamação

Mensagem: "Como faço para alterar o endereço de entrega?"
Intenção: dúvida

Mensagem: "Quero encerrar minha assinatura ainda este mês."
Intenção: cancelamento

Agora, classifique:

Mensagem: "Estou satisfeito com o serviço, mas quero cancelar porque vou mudar de país."
Intenção:
```

Aqui, a frase tem tom positivo mas intenção de “cancelamento”; os exemplos anteriores ajudam o modelo a priorizar a intenção explícita sobre o tom emocional.

### Exemplo 3: Classificação jurídica simplificada

```text
Você é um classificador de documentos jurídicos.
Classifique o texto em uma das categorias:
- contrato
- petição
- sentença
- parecer

Exemplos:

Texto: "Pelo presente instrumento particular de compra e venda, as partes abaixo assinadas..."
Classe: contrato

Texto: "Isto posto, julgo procedente o pedido formulado na petição inicial..."
Classe: sentença

Texto: "Trata-se de consulta sobre a possibilidade de acumulação de cargos públicos..."
Classe: parecer

Texto: "Excelentíssimo Senhor Doutor Juiz de Direito da 5ª Vara Cível..."
Classe: petição

Agora, classifique:

Texto: "Opino pela improcedência do pedido, nos termos da fundamentação acima."
Classe:
```

Os exemplos mostram ao modelo a estrutura típica de cada categoria. Mesmo que ele nunca tenha visto essas frases específicas durante o treinamento, o padrão “opino pela…” + linguagem técnica sugere “parecer”.

### Por que few-shot costuma funcionar melhor?

Em *few-shot*, o modelo não precisa só confiar no que “acha que significa” “reclamação”, “elogio” ou “parecer”; ele vê exemplos concretos de textos que o usuário está chamando assim e se adapta a esse padrão. Isso é particularmente útil quando:

- A definição das classes é específica de um projeto (por exemplo, “urgência alta” na sua empresa não é igual à de outra).
- O domínio tem jargão próprio (jurídico, médico, financeiro) e você pode colocar exemplos reais.
- Existem ambiguidades inerentes (textos com sentimentos mistos, intenções múltiplas) e os exemplos indicam como resolvê-las.

Porém, *few-shot* também tem limitações: o contexto do *prompt* é finito (não dá para colocar centenas de exemplos), e a escolha dos exemplos influencia muito o comportamento do modelo. Exemplos enviesados ou pouco representativos podem induzir classificações erradas em larga escala.


## Problemas comuns de prompting em classificação supervisionada

Ao usar *zero-shot* ou *few-shot* de forma “supervisionada” (por exemplo, para rotular um grande conjunto de dados ou para construir um sistema de produção), surgem alguns problemas típicos de *prompting*:

- **Ambiguidade nas instruções:** se as classes não são definidas de forma clara (“neutro” às vezes inclui misto, às vezes não), o modelo produz rótulos inconsistentes, o que atrapalha tanto supervisão humana quanto métricas de avaliação.
- **Desequilíbrio de exemplos no prompt:** em *few-shot*, se quase todos os exemplos são de uma classe, o modelo tende a repetir essa classe com mais frequência, mesmo quando o texto não justifica.
- **Saída ruidosa:** se o *prompt* não força um formato estrito (“responda apenas com…”), o modelo pode responder com frases completas e comentários, o que complica o pós-processamento.
- **Sensibilidade a detalhes superficiais:** mudar ligeiramente a redação (“diga” vs “classifique”) ou a ordem das classes pode alterar a distribuição de respostas, o que dificulta a reprodutibilidade e comparações justas entre variações.
- **Vazamento e enviesamento:** prompts que sugerem a resposta (“este texto parece ofensivo; diga se é ofensivo ou não”) podem induzir viés, tornando difícil separar o que é “conhecimento do modelo” do que é “sugestão do prompt”.

Em cenários supervisionados sérios (por exemplo, produção ou rotulagem de datasets), é importante tratar o *prompt* como parte do desenho experimental: documentar exatamente qual texto foi usado, testar variações, checar estabilidade das respostas e, sempre que possível, combinar *prompting* com uma camada de decisão mais controlada (como uma cabeça de classificação treinável).


## Cabeças de classificação treináveis em LLMs

Embora *zero-shot* e *few-shot* permitam usar LLMs como classificadores “só com texto”, muitas aplicações supervisionadas se beneficiam de acoplar cabeças de classificação treináveis às representações internas do modelo. A ideia é usar o *decoder* como extrator de características profundas e deixar a decisão final para uma camada supervisionada pequena, que pode ser treinada e avaliada com mais controle.

### Estratégia 1: cabeça sobre o estado final

Uma abordagem comum é construir um *prompt* padrão (por exemplo, “Texto: …\nClasse:”) e usar o vetor de estado do modelo na posição logo após “Classe:” como representação do par texto–instrução. Sobre esse vetor, adiciona-se uma camada linear com *softmax* com uma saída por classe. Durante o treinamento:

1. Apresenta-se o *prompt* com o texto e o espaço para a classe.
2. Extrai-se o vetor de estado na posição-alvo.
3. Passa-se esse vetor pela cabeça de classificação.
4. Calcula-se a perda (por exemplo, *cross-entropy* entre a distribuição prevista e o rótulo verdadeiro).
5. Atualizam-se os pesos da cabeça e, se desejado, uma parte dos pesos do LLM (ou apenas módulos leves, como LoRA).

Essa configuração aproxima o uso de LLMs de um classificador neural tradicional, mas com o benefício de que as representações internas já codificam muita informação semântica sobre o texto.

### Estratégia 2: cabeça sobre logits de tokens de classe

Outra estratégia é representar cada classe por um ou mais tokens do vocabulário (“positivo”, “negativo”, “neutro”) e olhar diretamente para os *logits* do modelo nesses tokens na posição de saída. Em vez de deixar o modelo escolher qualquer palavra do vocabulário como próxima saída, restringe-se a decisão a um conjunto de tokens que representam as classes.

Sobre esses *logits* restritos, é possível:

- Aplicar uma *softmax* apenas nas classes e usar isso diretamente como distribuição de probabilidade de classe.
- Treinar uma pequena camada de calibração (por exemplo, uma regressão logística) que pega os *logits* ou probabilidades e ajusta a distribuição final, o que pode melhorar a calibração e corrigir certos vieses.

Isso mantém o vínculo semântico entre classes e palavras reais do vocabulário, ao mesmo tempo em que introduz um componente supervisionado explícito para controle e avaliação.

### Vantagens das cabeças treináveis

Ao introduzir cabeças de classificação treináveis, ganhamos:

- **Estabilidade:** pequenas variações de *prompt* têm menos impacto na decisão final, pois a cabeça aprende um mapeamento mais robusto dos estados internos para as classes.
- **Avaliação clara:** podemos treinar e testar a cabeça com métricas tradicionais (acurácia, F1, AUC), independentemente da parte generativa do modelo.
- **Calibração:** é mais fácil calibrar probabilidades de saída, importante quando as decisões dependem de limiares (por exemplo, enviar para revisão humana apenas casos com probabilidade de “tóxico” acima de 0,8).
- **Hibridização:** permite usar o LLM tanto como gerador aberto (para explicações, resumos) quanto como backend de um classificador supervisionado, compondo diferentes módulos em um mesmo sistema.

Na prática, muitos sistemas modernos combinam as duas ideias: usam *prompting* cuidadoso para alinhar o comportamento do LLM e, em tarefas de classificação críticas, acoplam uma cabeça supervisionada (eventualmente com *fine-tuning* leve como LoRA) para obter decisões mais previsíveis, auditáveis e ajustadas ao domínio específico.

