#!/bin/bash

# Ativa proxy para o pip
PROXY="http://ps:lions@192.168.10.1:3128"

# Cria o ambiente virtual, se ainda não existir
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi

# Ativa o ambiente virtual
source venv/bin/activate

# Atualiza pip (opcional, mas recomendado)
python -m pip install --upgrade pip --proxy="$PROXY"

# Instala dependências do requirements.txt
echo "Instalando dependências..."
python -m pip install -r requirements.txt --proxy="$PROXY"

echo "✅ Ambiente pronto! Para ativar, use: source venv/bin/activate"
