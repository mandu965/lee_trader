FROM node:20

WORKDIR /app

# This Dockerfile is built with `node/` as the build context.
COPY package*.json ./
RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY . .

EXPOSE 3000
CMD ["npm", "run", "start"]
