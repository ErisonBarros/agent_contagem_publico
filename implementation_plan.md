# Goal Description
Traduzir as descrições de todos os comandos do framework GSD (`gsd-*`) no diretório de skills (`~/.gemini/antigravity/skills/`) para o português do Brasil, utilizando um script Python para realizar a modificação em massa, de forma segura e garantindo o formato UTF-8.

## User Review Required
> [!IMPORTANT]
> A modificação em massa vai alterar todos os arquivos `SKILL.md` das skills `gsd-*` no diretório do sistema. O script utilizará um dicionário traduzido antecipadamente para evitar dependência de APIs externas e garantir máxima precisão técnica. 
> Por favor, confirme se podemos prosseguir com a criação e execução deste script Python.

## Proposed Changes
Criar um script Python na pasta de trabalho que:
1. Iterará sobre todos os diretórios `gsd-*` no caminho local `C:\Users\USUARIO\.gemini\antigravity\skills\`.
2. Abrirá cada arquivo `SKILL.md` garantindo a leitura e gravação em UTF-8.
3. Localizará a linha `description: ...` no cabeçalho YAML.
4. Substituirá a descrição em inglês pela tradução profissional e fiel em português, preservando a sintaxe original do arquivo YAML.
5. Manterá intactos os nomes dos comandos e o restante do conteúdo do arquivo.

### Script de Tradução
O script `translate_gsd_skills.py` incluirá o mapeamento exato de todas as cerca de 65 skills `gsd-*` e a lógica de substituição de texto via expressões regulares (Regex).

## Verification Plan
### Automated Tests
- Executar o script no terminal.
- Verificar a saída do script para garantir que os arquivos foram processados e alterados com sucesso.

### Manual Verification
- Visualizar aleatoriamente alguns arquivos (ex: `gsd-fast/SKILL.md`, `gsd-progress/SKILL.md`) para garantir que o cabeçalho YAML possui a descrição em português, mantendo a estrutura original intacta.

- **Erison Barros**

[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--4879--6880-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-4879-6880)
