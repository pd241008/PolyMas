.PHONY: all build proto clean test lint docker-up docker-down

# ---------- Toolchain versions ----------
RUST_VERSION  := 1.78
GO_VERSION     := 1.22
SCALA_VERSION  := 3.4
PYTHON_VERSION := 3.12
NODE_VERSION   := 20

# ---------- Protobuf ----------
PROTO_DIR    := proto
PROTO_OUT    := proto/gen

PROTO_OUT    := proto/gen
PROTO_DIR    := proto

proto:
	@echo "==> Generating protobuf code..."
	@mkdir -p $(PROTO_OUT)/go $(PROTO_OUT)/python $(PROTO_OUT)/java
	PROTOC := protoc-25
	$$PROTOC --proto_path=$(PROTO_DIR) \
		--go_out=$(PROTO_OUT)/go --go_opt=paths=source_relative \
		--go-grpc_out=$(PROTO_OUT)/go --go-grpc_opt=paths=source_relative \
		--python_out=$(PROTO_OUT)/python \
		--grpc_python_out=$(PROTO_OUT)/python \
		--java_out=$(PROTO_OUT)/java \
		$(PROTO_DIR)/polymas/v1/*.proto
	$$PROTOC --proto_path=$(PROTO_DIR) \
		--grpc-java_out=$(PROTO_OUT)/java \
		$(PROTO_DIR)/polymas/v1/services.proto
	@echo "==> Protobuf generation complete."

# ---------- Scala ----------
.PHONY: build-scala
build-scala:
	@echo "==> Building Scala ingestion service..."
	cd services/ingestion-scala && sbt compile

.PHONY: test-scala
test-scala:
	cd services/ingestion-scala && sbt test

# ---------- Go ----------
.PHONY: build-go
build-go:
	@echo "==> Building Go normalization service..."
	cd services/normalization-go && go build -o bin/server ./cmd/server

.PHONY: test-go
test-go:
	cd services/normalization-go && go test ./...

.PHONY: lint-go
lint-go:
	cd services/normalization-go && go vet ./...

# ---------- Rust ----------
.PHONY: build-rust
build-rust:
	@echo "==> Building Rust control plane..."
	cd services/control-plane-rust && cargo build --release

.PHONY: test-rust
test-rust:
	cd services/control-plane-rust && cargo test

.PHONY: lint-rust
lint-rust:
	cd services/control-plane-rust && cargo clippy -- -D warnings

# ---------- Python ----------
.PHONY: setup-python
setup-python:
	@echo "==> Setting up Python ML engine venv..."
	cd services/ml-engine-python && python3 -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt && \
		.venv/bin/pip install pytest

.PHONY: test-python
test-python:
	cd services/ml-engine-python && .venv/bin/pytest tests/ -v

.PHONY: build-dataset
build-dataset:
	cd services/ml-engine-python && PYTHONPATH=. .venv/bin/python scripts/build_dataset.py

.PHONY: lint-python
lint-python:
	cd services/ml-engine-python && .venv/bin/ruff check . && \
		.venv/bin/mypy polymas_ml/

# ---------- Next.js ----------
.PHONY: install-dashboard
install-dashboard:
	@echo "==> Installing Next.js dashboard deps..."
	cd apps/dashboard-nextjs && npm install

.PHONY: build-dashboard
build-dashboard:
	cd apps/dashboard-nextjs && npm run build

.PHONY: dev-dashboard
dev-dashboard:
	cd apps/dashboard-nextjs && npm run dev

# ---------- Docker ----------
.PHONY: docker-up
docker-up:
	@echo "==> Starting all services via Docker Compose..."
	docker compose up --build -d

.PHONY: docker-down
docker-down:
	docker compose down

.PHONY: docker-logs
docker-logs:
	docker compose logs -f

# ---------- Aggregate targets ----------
.PHONY: build
build: proto build-scala build-go build-rust build-dashboard

.PHONY: test
test: test-scala test-go test-rust test-python

.PHONY: lint
lint: lint-go lint-rust lint-python

.PHONY: clean
clean:
	cd services/control-plane-rust && cargo clean
	cd services/normalization-go && rm -rf bin/
	cd services/ingestion-scala && sbt clean
	cd services/ml-engine-python && rm -rf .venv __pycache__ .pytest_cache
	cd apps/dashboard-nextjs && rm -rf .next out node_modules
	rm -rf $(PROTO_OUT)

.PHONY: all
all: proto build test

.DEFAULT_GOAL := all
