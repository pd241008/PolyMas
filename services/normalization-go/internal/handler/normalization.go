package handler

import (
	"context"
	"log"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// NormalizationService implements the gRPC normalization service.
type NormalizationService struct {
	// TODO: embed unimplemented server once protobuf is generated
}

func NewNormalizationService() *NormalizationService {
	return &NormalizationService{}
}

// NormalizeBatch cleans and normalizes raw payloads into canonical PatientProfile.
func (s *NormalizationService) NormalizeBatch(ctx context.Context, batchID string, payloads [][]byte) error {
	log.Printf("normalizing batch %s with %d payloads", batchID, len(payloads))

	// TODO: Implement
	// 1. Parse raw JSON bytes per source format (GWAS vs ImmPort)
	// 2. Map to canonical PatientProfile proto
	// 3. Run validation rules
	// 4. Return accepted/rejected counts

	return status.Errorf(codes.Unimplemented, "NormalizeBatch not yet implemented")
}

// ValidateProfile checks a single PatientProfile against schema rules.
func (s *NormalizationService) ValidateProfile(ctx context.Context) error {
	// TODO: Implement validation logic
	// - Required fields check
	// - Enum bounds validation
	// - Score range validation (0.0 - 1.0 for normalized scores)
	return status.Errorf(codes.Unimplemented, "ValidateProfile not yet implemented")
}
