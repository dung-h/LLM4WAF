FROM caddy:builder AS builder

RUN xcaddy build \
    --with github.com/corazawaf/coraza-caddy/v2

FROM caddy:latest

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
