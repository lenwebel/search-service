FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cargo build --release --bin search-service

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/search-service /usr/local/bin/
COPY schemas /app/schemas
WORKDIR /app
# Segment cache on local disk: App Service's /home is a network share.
ENV BIND=0.0.0.0:8080 SEGMENT_CACHE_DIR=/tmp/segments
EXPOSE 8080
CMD ["search-service", "serve"]
