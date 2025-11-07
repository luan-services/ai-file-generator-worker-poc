# Dependências de GPU

Essas dependências foram instaladas para permitir que o demucs utilize a GPU com suporte para nvidia versão 30xx / 40xx no Python 3.13.1 (/cu126), evitando uso da CPU.

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec
```


# Conceito de CPU vs. GPU

Não é 100% automático. Precisa fazer duas coisas:

- Instalar o PyTorch correto (o que você acabou de fazer).

- Mandar o script usar a GPU (isso é feito no código).

# Como o PyTorch Decide?

Por padrão, o PyTorch sempre coloca tudo na CPU. A CPU é o "dispositivo" (device) padrão.

Se tivesse instalado a versão CPU-only (ou se o computador não tiver uma placa NVIDIA), o PyTorch nem tenta procurar uma GPU. Ele faria todos os cálculos na CPU (o que seria muito mais lento para treinar um modelo).

Se essa versão fosse instalada, o pychorch usaria por padrão a CPU.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Como foi instalado a versão cu126, o PyTorch é capaz "habilidade" de falar com a GPU. O pyTorch vai procurar por ela automáticamente. (Também é possível forçar o uso)

# Dependências de library 

Para o test_v1, é necessário as seguintes dependências:

```bash
pip install demucs librosa
```

Isso torna o código capaz de separar áudio em instrumentos (demucs), e de gerar bpm (librosa).

# FFMPEG e CHOCO

Fiz a instalação do pacote ffmpeg, um jeito simples é com o instalador de pacotes choco, no powershell rode:

```bash
Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = `
[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Em seguida:

```bash
choco install ffmpeg -y
```

NOTA: Removi o ffmpeg v8 e fiz a instalação manual da v7, pq a versão do torch que instalei não é compatível com o ffmpeg 8. (não dá com choco).

# Separação de Arquivos

## t-1.py

Responsável por checar o funcionamento do demucs para separar sons e do librosa para pegar a bpm (média geral) da música (usando a música original). 

No próximo poc, é necessário mudar o librosa p usar o arquivo drums.wav gerado, caso a música não tenha nada
em drums, é preciso ser capaz de reconhecer isso e usar other.wav. (voz, piano, guitarra, etc).

Também é necessário que o librosa seja capaz de pegar o bpm continuo, ex: segundo 1-20 - 80bpm / segundo 21-30 - 60bpm / segundo 31-120-fim - 80bpm