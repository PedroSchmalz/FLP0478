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

# Instalação do Software

## O que é o Google Colab?

O Google Colab é uma ferramenta gratuita baseada na nuvem que permite executar código Python diretamente no navegador. Ele é amplamente utilizado para aprendizado de máquina, análise de dados e outras tarefas que exigem Python.

---

## Como acessar o Google Colab?

1. Acesse o site do Google Colab: [Google Colab](https://colab.research.google.com/).
2. Faça login com sua conta do Google.

---

## Criando um novo notebook

1. Clique em **"File"** (Arquivo) > **"New Notebook"** (Novo Notebook).
2. Um novo notebook será aberto com uma célula pronta para receber código.

---

## Estrutura do Notebook

- **Células de Código**: Para escrever e executar código Python.
- **Células de Texto**: Para adicionar explicações ou formatações em Markdown.

---

## Executando Código

1. Escreva o código em uma célula de código, por exemplo:
   ```python
   print("Olá, Google Colab!")
   ```
2. Pressione `Shift + Enter` ou clique no botão de "play" ao lado da célula para executar.

---

## Upload de Arquivos

1. Clique no ícone de pasta no lado esquerdo.
2. Clique em **"Upload"** para carregar arquivos do seu computador.

---

## Conectando ao Google Drive

1. Execute o seguinte código para montar o Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Siga as instruções para autorizar o acesso.

---

## Instalando Bibliotecas

Use o comando `!pip install` para instalar bibliotecas diretamente no Colab. Por exemplo:
```python
!pip install pandas
```

---

## Salvando e Compartilhando

1. O notebook é salvo automaticamente no Google Drive.
2. Para compartilhar, clique em **"Share"** (Compartilhar) no canto superior direito e configure as permissões.

---

## Recursos Adicionais

- [Documentação Oficial do Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [Vídeo Tutorial](https://www.youtube.com/embed/UCb-b82tzLo?)

```{video} https://www.youtube.com/embed/UCb-b82tzLo?
```

