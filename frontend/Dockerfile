FROM node:20-alpine AS base

WORKDIR /app

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# ============================================
# Development stage - runs Vite dev server with HMR
# ============================================
FROM base AS development

COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# ============================================
# Production stage - builds static files, serves with nginx
# ============================================
FROM base AS builder

COPY . .
RUN npm run build

FROM nginx:alpine AS production
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# install curl for health checks
RUN apk add --no-cache curl

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]