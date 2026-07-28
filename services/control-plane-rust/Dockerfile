FROM rust:1.78-slim AS build

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release 2>/dev/null || true
RUN rm -rf src

COPY src/ src/
COPY build.rs .
COPY ../../proto/ /proto/
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/target/release/polymas-control-plane .
EXPOSE 50053
ENTRYPOINT ["./polymas-control-plane"]
