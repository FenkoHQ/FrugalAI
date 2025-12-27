# Build stage
FROM golang:1.25-alpine AS builder

WORKDIR /build

# Install git for go mod download
RUN apk add --no-cache git

# Copy go mod and sum files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -ldflags="-s -w" -o frugalai ./cmd/frugalai

# Final stage - minimal Alpine image
FROM alpine:3.19

# Install ca-certificates for HTTPS and dumb-init for proper signal handling
RUN apk --no-cache add ca-certificates dumb-init

# Create non-root user
RUN addgroup -g 1000 appgroup && \
    adduser -u 1000 -G appgroup -s /bin/sh -D appuser

# Copy binary from builder
COPY --from=builder /build/frugalai /usr/local/bin/frugalai

# Switch to non-root user
USER appuser

# Expose default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
ENTRYPOINT ["dumb-init", "--"]
CMD ["frugalai"]
