# Dados utilizados

## Fonte
- **Base:** Dados do SP156 – Solicitações (3º trimestre de 2026)
- **Origem:** Portal de Dados Abertos da Cidade de São Paulo (Prefeitura de São Paulo)
- **Link:** https://dados.prefeitura.sp.gov.br/dataset/dados-do-sp156/resource/4288a1e6-d495-42d6-be1e-6169f729cced
- **Período coberto:** 01/07/2026 a 30/09/2026 (data de abertura da solicitação)
- **Natureza:** dados reais e públicos de solicitações feitas por cidadãos ao SP156. Nenhum dado simulado foi usado.

## Como o arquivo foi obtido
O arquivo original (Excel) foi baixado no Google Colab e filtrado pelo tema **"Lixo e limpeza"**, gerando o `dados_lixo_e_limpeza.csv` (57.776 linhas e 15 colunas).

```python
df_filtrado = df[df['Tema'] == 'Lixo e limpeza']
df_filtrado.to_csv('dados_lixo_e_limpeza.csv', index=False, encoding='utf-8-sig')
```

## Colunas do arquivo
Data de Abertura, Data de Finalização, Meio de Comunicação, Tema, Assunto, Serviço, Status, Órgão, Logradouro, Número, CEP, Latitude, Longitude, Prefeitura Operacional, Distrito.

## Tratamento aplicado no notebook
1. Remoção das solicitações com status "Cancelada".
2. Remoção das linhas sem "Prefeitura Operacional" (necessária para prever por região).
3. Manutenção apenas das 32 Prefeituras Regionais com 300 ou mais solicitações no período.
4. Agregação em uma tabela diária (dia × região), com zero nos dias sem solicitações.
5. Remoção, apenas do conjunto de treino, de 9 picos atípicos (regra do IQR por região).

Os detalhes e o raciocínio de cada etapa estão no notebook `notebook_coletacerta.ipynb`.
