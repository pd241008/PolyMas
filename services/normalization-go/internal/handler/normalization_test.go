package handler

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestNewNormalizationService(t *testing.T) {
	svc := NewNormalizationService()
	if svc == nil {
		t.Error("NewNormalizationService() returned nil")
	}
}

func TestNormalizeBatchUnimplemented(t *testing.T) {
	svc := NewNormalizationService()
	err := svc.NormalizeBatch(context.Background(), "b1", [][]byte{})
	if err == nil {
		t.Fatal("NormalizeBatch() expected error, got nil")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %T", err)
	}
	if st.Code() != codes.Unimplemented {
		t.Errorf("NormalizeBatch() code = %v, want Unimplemented", st.Code())
	}
}

func TestValidateProfileUnimplemented(t *testing.T) {
	svc := NewNormalizationService()
	err := svc.ValidateProfile(context.Background())
	if err == nil {
		t.Fatal("ValidateProfile() expected error, got nil")
	}

	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %T", err)
	}
	if st.Code() != codes.Unimplemented {
		t.Errorf("ValidateProfile() code = %v, want Unimplemented", st.Code())
	}
}
