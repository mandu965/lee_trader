FROM node:20

ARG NODE_VERSION=20.19.5
WORKDIR /app

# lightweight deps (pg is pure JS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY . .

EXPOSE 3000
CMD ["npm", "run", "start"]
