# mindspore-playground



# Usando Docker

A forma mais simples de usar o minspore é através do container docker seguindo o passo a passo descrito abaixo: 

Descrito em: https://mindspore.cn/install/en
![image](https://user-images.githubusercontent.com/276077/212794989-a8f5e5d0-3cab-4af3-9b12-856d9ece97fd.png)

## 1. Baixar a imagem do docker

```bash
sudo docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:1.9.0
```

Nesse comando será instalado o mindspore na versão 1.9.9

## 2. Acesso ao container
```bash
sudo docker run -it swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:1.9.0 /bin/bash
```

## 3. Testando um código dentro do container

```bash
python -c "import mindspore;mindspore.run_check()"
```


## 4. Iniciando o container com docker-compose
```bash
sudo docker-compose up
```

## 5. Finalizando o container com docker-compose
```bash
sudo docker-compose down
```

Os arquivos copy_into.sh e copy_execute.sh são scripts que permitem copiar arquivos para dentro do container e também executar.

# Usando Pip

## Criar ambiente com o Anaconta
```bash
conda create -n py39 python=3.9 anaconda
```

## Inicializar o ambiente
```bash
conda activate py39
```

## Instalar a versão do Mindspore desejada
```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/cpu/x86_64/mindspore-1.9.0-cp39-cp39-linux_x86_64.whl
```
Link do pacote para o mindspore 1.9.0 da plataforma Linux x86_64. Outros pacotes podem ser obtidos em: https://www.mindspore.cn/versions/en
