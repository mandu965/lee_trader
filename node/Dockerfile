FROM node:20

WORKDIR /app

# This Dockerfile is built with the repository root as the build context.
COPY node/package*.json ./
RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY node/. .

EXPOSE 3000
CMD ["npm", "run", "start"]
