FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"

RUN conda create -y -n ViT python=3.10
RUN conda run -n ViT pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git /app/Dassl.pytorch
RUN conda run -n ViT pip install -r /app/Dassl.pytorch/requirements.txt
RUN conda run -n ViT bash -lc "cd /app/Dassl.pytorch && pip install --no-build-isolation -e ."

COPY . /app/ViT
WORKDIR /app/ViT

RUN conda run -n ViT bash -lc "if [ -f requirements.txt ]; then pip install -r requirements.txt; fi"

ENV CONDA_DEFAULT_ENV=ViT
ENV PATH="/opt/conda/envs/ViT/bin:$PATH"

CMD ["/bin/bash"]
