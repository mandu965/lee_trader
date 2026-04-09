FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install --legacy-peer-deps \
  && npm cache clean --force

COPY . .

EXPOSE 3000
CMD ["npm", "run", "start"]
