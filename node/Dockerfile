FROM node:20

WORKDIR /app

# Render deploys this Dockerfile with the repo root as build context.
COPY node/package*.json ./
RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY node/. .

EXPOSE 3000
CMD ["npm", "run", "start"]
