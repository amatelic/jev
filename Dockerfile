FROM node:22-slim

# Model selection at build time (or override at runtime with -e JEV_MODEL=...)
ARG JEV_MODEL=kev-0.6b
ENV JEV_MODEL=${JEV_MODEL}
ENV JEV_DTYPE=q4
ENV JEV_PORT=9401

WORKDIR /app

# Install dependencies first (cached layer)
COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund

# Copy source and build
COPY . .
RUN npm run build

# Pre-download the model if specified (optional: skip for smaller images)
# RUN node -e "import('open-jev').then(m => m.OpenJev.load({ model: process.env.JEV_MODEL }))"

EXPOSE 9401

CMD ["node", "dist/server.js"]
