FROM nginx:1.24.0

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc-dev \
        make \
        openssl \
        libssl-dev \
        libpcre3-dev \
        zlib1g-dev \
        git \
        wget

# Download Nginx and NAXSI
WORKDIR /usr/src
RUN wget http://nginx.org/download/nginx-1.24.0.tar.gz && tar -zxvf nginx-1.24.0.tar.gz
RUN git clone https://github.com/nbs-system/naxsi.git

# Compile module
WORKDIR /usr/src/nginx-1.24.0
RUN ./configure --with-compat --add-dynamic-module=../naxsi/naxsi_src && make modules

# Move module
RUN mkdir -p /etc/nginx/modules
RUN cp objs/ngx_http_naxsi_module.so /etc/nginx/modules/

# Load module in main config
RUN sed -i '1iload_module modules/ngx_http_naxsi_module.so;' /etc/nginx/nginx.conf
