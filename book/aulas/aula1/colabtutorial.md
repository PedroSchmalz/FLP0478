---

jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Software

Na disciplina, trabalharemos com o Python usando o `Jupyter Notebook`, que contém arquivos no formato `.ipynb`. Esses notebooks permitem combinar código, texto explicativo, visualizações e resultados em um único documento interativo. Eles são amplamente utilizados em ciência de dados, aprendizado de máquina e ensino de programação, pois oferecem uma maneira prática de executar e documentar análises de forma integrada. Durante o curso, utilizaremos tanto o **Google Colab**, uma ferramenta baseada na nuvem que nos permite usar *jupyter notebooks* no *browser*, quanto o **VS Code**, uma alternativa local, para trabalhar com esses notebooks.

`````{tab-set}
````{tab-item} Google Colab (Preferido)

## Tutorial Google Colab


Para quem prefere trabalhar com a nuvem, a opção é o Google Colab. E este aplicativo será a preferência em nossas aulas, tanto para evitar problemas de compatibilidade com pacotes e funções, quanto para facilitar para aqueles que possuem máquinas menos "potentes". O Google Colab é uma ferramenta gratuita em nuvem que permite executar código Python diretamente no navegador. Ele é amplamente utilizado para aprendizado de máquina, análise de dados e outras tarefas que exigem Python.

---

### Como acessar o Google Colab?

1. Acesse o site do Google Colab: [Google Colab](https://colab.research.google.com/).
2. Faça login com sua conta do Google.


---

### Criando um novo notebook

1. Clique em **"File"** (Arquivo) > **"New Notebook"** (Novo Notebook).
2. Um novo notebook será aberto com uma célula pronta para receber código.

```{figure} ../aula1/images/colab4.png
---
width: 100%
name: colab4
align: center
---

```

---

### Estrutura do Notebook

- **Células de Código**: Para escrever e executar código Python.
- **Células de Texto**: Para adicionar explicações ou formatações em Markdown.

---

### Executando Código

1. Escreva o código em uma célula de código, por exemplo:
   ```python
   print("Olá, Google Colab!")
   ```
2. Pressione `Shift + Enter` ou clique no botão de "play" ao lado da célula para executar.

---

### Upload de Arquivos

1. Clique no ícone de pasta no lado esquerdo.
2. Clique em **"Upload"** para carregar arquivos do seu computador.

---

### Pegando arquivos do Google Drive

1. Execute o seguinte código para montar o Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Siga as instruções para autorizar o acesso.


### Usando diretamente no Google Drive

1. Abra o Google Drive no seu *browser* e clique com o botão direito do mouse em algum arquivo no formato `.ipynb`
2. Clique em "Abrir com" > "Google Colaboratory"

```{figure} ../aula1/images/colab1.png
---
width: 100%
name: colab1
align: center
---

```

2.a. Talvez a sua conta do Drive ainda não tenha o Colab instalado. Se esse for o seu caso, clique em "Abrir com" > "Conectar mais apps".

```{figure} ../aula1/images/colab2.png
---
width: 100%
name: colab2
align: center
---

```

2.b. Procure por "Colab" ou "Google Colaboratory" na barra de pesquisa.

```{figure} ../aula1/images/colab3.png
---
width: 100%
name: colab3
align: center
---

```

2.c. Instale o Colab e tente o passo 2 novamente.



---

### Instalando Bibliotecas

Use o comando `!pip install` para instalar bibliotecas diretamente no Colab. Por exemplo:
```python
!pip install pandas
```

---

### Salvando e Compartilhando

1. O notebook é salvo automaticamente no Google Drive.
2. Para compartilhar, clique em **"Share"** (Compartilhar) no canto superior direito e configure as permissões.
3. Você também pode baixar o arquivo `.ipynb` para compartilhar com colegas ou usar localmente no seu computador, clicando em "Arquivo" > "Fazer Download" > "Baixar o .ipynb"

```{figure} ../aula1/images/colab5.png
---
width: 100%
name: colab5
align: center
---

```



````

````{tab-item} VS Code

## Tutorial VS Code

Embora o Google Colab seja uma excelente ferramenta baseada na nuvem, você também pode usar o **Visual Studio Code (VS Code)** para trabalhar com arquivos `.ipynb` localmente. Isso pode ser útil se você preferir um ambiente offline ou mais personalizável. No entanto, você terá que tomar mais cuidado com a instalação de pacotes, versão destes pacotes e do Python, localização dos arquivos, etc.

---

### Passo 1: Instalar o VS Code

1. Baixe o Visual Studio Code no site oficial: [VS Code](https://code.visualstudio.com/).
2. Instale o VS Code seguindo as instruções para o seu sistema operacional.

---

### Passo 2: Instalar a Extensão para Jupyter

1. Abra o VS Code.
2. Vá até a aba de extensões clicando no ícone de quadrado no lado esquerdo ou pressione `Ctrl + Shift + X`.
3. Pesquise por **"Jupyter"** e clique em **Install** para instalar a extensão oficial da Microsoft.

```{figure} ../aula1/images/vscode1.png
---
width: 100%
name: vscode1
align: center
---

```

---

### Passo 3: Configurar o Python no VS Code

1. Certifique-se de que o Python está instalado no seu computador. Caso não esteja, baixe-o em [Python.org](https://www.python.org/).

```{figure} ../aula1/images/python.png
---
width: 100%
name: python
align: center
---

```


2. Instale a extensão **Python** no VS Code (siga o mesmo processo da extensão Jupyter).
3. Configure o interpretador Python abrindo um arquivo (ou criando) `.ipynb` e selecionando o ambiente Python desejado.

```{figure} ../aula1/images/vscode2.png
---
width: 100%
name: vscode2
align: center
---

```

```{figure} ../aula1/images/vscode3.png
---
width: 100%
name: vscode3
align: center
---

```

4. Cheque o seguinte [tutorial](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) do VScode se tiver algum problema nos passos anteriores.



---

### Passo 4: Abrir e Editar Arquivos `.ipynb`

1. Abra o arquivo `.ipynb` no VS Code clicando em **File** > **Open File** e selecionando o arquivo.
2. O arquivo será aberto em uma interface interativa semelhante ao Jupyter Notebook.
3. Execute as células clicando no botão de "play" ao lado de cada célula ou pressionando `Shift + Enter`.


````
`````

### Alternar entre Colab e VS Code

- Você pode alternar entre o Google Colab e o VS Code facilmente:
  - **Do Colab para o VS Code**: Baixe o notebook no Colab clicando em **File** > **Download** > **Notebook (.ipynb)** e abra no VS Code.
  - **Do VS Code para o Colab**: Faça upload do arquivo `.ipynb` no Colab clicando em **File** > **Upload Notebook**.


## Recursos Adicionais

- [Documentação do VS Code para Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Documentação Oficial do Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)