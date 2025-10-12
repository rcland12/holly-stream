FROM nginx:1.26-alpine AS builder

ARG NGINX_VERSION=1.26.3
ARG RTMP_MODULE_VERSION=1.2.2
ARG NGINX_USER=nginx

RUN apk add --no-cache --virtual .build-deps \
        gcc \
        git \
        libc-dev \
        linux-headers \
        make \
        openssl-dev \
        pcre-dev \
        wget \
        zlib-dev && \
    wget -O nginx.tar.gz http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz && \
    tar -xzf nginx.tar.gz && \
    git clone --depth 1 --branch v${RTMP_MODULE_VERSION} https://github.com/arut/nginx-rtmp-module.git

WORKDIR /nginx-${NGINX_VERSION}
RUN ./configure \
        --prefix=/etc/nginx \
        --sbin-path=/usr/sbin/nginx \
        --modules-path=/usr/lib/nginx/modules \
        --conf-path=/etc/nginx/nginx.conf \
        --error-log-path=/var/log/nginx/error.log \
        --http-log-path=/var/log/nginx/access.log \
        --pid-path=/var/run/nginx.pid \
        --lock-path=/var/run/nginx.lock \
        --http-client-body-temp-path=/var/cache/nginx/client_temp \
        --http-proxy-temp-path=/var/cache/nginx/proxy_temp \
        --http-fastcgi-temp-path=/var/cache/nginx/fastcgi_temp \
        --http-uwsgi-temp-path=/var/cache/nginx/proxy_temp \
        --http-scgi-temp-path=/var/cache/nginx/scgi_temp \
        --user=${NGINX_USER} \
        --group=${NGINX_USER} \
        --with-compat \
        --with-file-aio \
        --with-threads \
        --with-http_dav_module \
        --with-http_gzip_static_module \
        --with-http_realip_module \
        --with-http_ssl_module \
        --with-http_v2_module \
        --with-cc-opt='-Os -fstack-clash-protection -Wformat -Werror=format-security -fno-plt -g' \
        --with-ld-opt='-Wl,--as-needed,-O1,--sort-common -Wl,-z,pack-relative-relocs' \
        --add-dynamic-module=../nginx-rtmp-module && \
    make modules

FROM nginx:1.26-alpine

ARG NGINX_USER=nginx

COPY --from=builder /nginx-${NGINX_VERSION}/objs/ngx_rtmp_module.so /usr/lib/nginx/modules/

RUN apk add --no-cache \
        ca-certificates \
        openssl \
        pcre \
        zlib && \
    mkdir -p \
        /var/cache/nginx/client_temp \
        /var/cache/nginx/proxy_temp \
        /var/cache/nginx/fastcgi_temp \
        /var/cache/nginx/uwsgi_temp \
        /var/cache/nginx/scgi_temp \
        /var/log/nginx \
        /tmp/hls \
        /var/www/html/stream && \
    chown -R ${NGINX_USER}:${NGINX_USER} \
        /var/cache/nginx \
        /var/log/nginx \
        /tmp/hls \
        /var/www/html/stream

COPY --chown=${NGINX_USER}:${NGINX_USER} ./nginx.conf /etc/nginx/nginx.conf
COPY --chown=${NGINX_USER}:${NGINX_USER} ./stream /var/www/html/stream

USER ${NGINX_USER}

CMD ["nginx", "-g", "daemon off;"]
