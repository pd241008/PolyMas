package main

import (
	"os"
	"testing"
)

func TestDefaultPort(t *testing.T) {
	os.Unsetenv("GRPC_PORT")
	port := getEnvOrDefault("GRPC_PORT", "50052")
	if port != "50052" {
		t.Errorf("default port = %s, want 50052", port)
	}
}

func TestCustomPort(t *testing.T) {
	os.Setenv("GRPC_PORT", "50053")
	defer os.Unsetenv("GRPC_PORT")
	port := getEnvOrDefault("GRPC_PORT", "50052")
	if port != "50053" {
		t.Errorf("custom port = %s, want 50053", port)
	}
}
