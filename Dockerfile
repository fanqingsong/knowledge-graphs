FROM python:3.12-slim
USER root

# --- Use Huawei Cloud mirrors for apt + pip (much faster inside CN networks) ---
# Debian Trixie stores sources in deb822 format at /etc/apt/sources.list.d/debian.sources
RUN sed -i 's|http://deb.debian.org|https://repo.huaweicloud.com|g; s|http://security.debian.org|https://repo.huaweicloud.com|g' \
        /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|http://deb.debian.org|https://repo.huaweicloud.com|g; s|http://security.debian.org|https://repo.huaweicloud.com|g' \
        /etc/apt/sources.list

RUN apt-get update && apt-get install -y libmagic1

RUN mkdir -p /knowledge-graphs
WORKDIR /knowledge-graphs

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    -i https://repo.huaweicloud.com/repository/pypi/simple
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://repo.huaweicloud.com/repository/pypi/simple

COPY graphrag ./graphrag
COPY streamlit_pages ./streamlit_pages
COPY app.py ./app.py
COPY config_example.env ./config_example.env

EXPOSE 8080

# --server.fileWatcherType=poll makes Streamlit reliably detect file changes
# inside a bind-mounted volume (Docker Desktop / WSL2 / network filesystems).
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8080", \
            "--server.address=0.0.0.0", \
            "--server.fileWatcherType=poll", \
            "--server.runOnSave=true"]