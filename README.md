# mindspore-playground


# Instalação

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
