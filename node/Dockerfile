FROM node:20

ARG NODE_VERSION=20.19.5
WORKDIR /app

# system deps for native modules (better-sqlite3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-distutils \
    pkg-config \
    libsqlite3-dev \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
# allow self-signed certs when fetching node headers/prebuilds
RUN npm config set strict-ssl false
ENV NODE_TLS_REJECT_UNAUTHORIZED=0 \
    npm_config_strict_ssl=false \
    npm_config_build_from_source=true \
    npm_config_python=python3

# pre-download node headers to avoid TLS MITM during build
RUN mkdir -p /tmp/node-headers && \
    curl -k -L https://nodejs.org/download/release/v${NODE_VERSION}/node-v${NODE_VERSION}-headers.tar.gz \
      | tar xz -C /tmp/node-headers
ENV npm_config_nodedir=/tmp/node-headers/node-v${NODE_VERSION}

RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY . .

EXPOSE 3000
CMD ["npm", "run", "start"]
