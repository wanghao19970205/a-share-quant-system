# 在 Linux x86_64 环境运行本应用（含银河 AmazingData 券商 SDK）。
# 券商 SDK 仅提供 Linux x86_64 原生库，故 arm64 Mac 需用 docker build --platform linux/amd64 构建。
# 基础镜像走国内加速源（Docker Hub 在国内常超时）。
# 用 Python 3.12：AmazingData 依赖 scipy>=1.15.1，而 scipy 已停止支持 3.9（需 >=3.10）。
FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

# pip 也走国内源，避免 PyPI 超时（清华源较稳定）
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

# tgw 原生库运行所需的系统依赖（libgssapi_krb5 是 tgw .so 的依赖）
# apt 也走清华源：deb.debian.org 在这台机器上实测约 15KB/s。
RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libstdc++6 ca-certificates \
        libgssapi-krb5-2 libkrb5-3 \
    && rm -rf /var/lib/apt/lists/*

# 先装 Web 运行依赖（利用缓存）。
# 注意：容器只跑 Streamlit UI + 行情 + 读取量化预测 parquet，不做模型训练，
# 因此这里用精简的 requirements-web.txt，剔除 torch/xgboost/lightgbm/scikit-learn/tables
# 等训练侧重依赖，避免 build 时下载 torch(526MB) 等导致超时。训练在宿主机用 requirements.txt。
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 安装券商 SDK（Linux x86_64 wheel，放在 sdk/ 目录；AmazingData 用 cp312 匹配 Python 3.12）
COPY sdk/tgw-1.0.8.7-py3-none-any.whl sdk/AmazingData-1.1.7-cp312-none-any.whl /tmp/sdk/
RUN pip install --no-cache-dir /tmp/sdk/tgw-1.0.8.7-py3-none-any.whl \
    && pip install --no-cache-dir /tmp/sdk/AmazingData-1.1.7-cp312-none-any.whl

# 让 tgw 的原生 .so 可被动态加载
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/tgw/common_linux_lib64

# 应用代码
COPY app.py mobile_app.py ./
COPY stock_analyzer ./stock_analyzer
COPY .streamlit ./.streamlit

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
