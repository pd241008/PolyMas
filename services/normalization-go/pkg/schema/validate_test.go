package schema

import "testing"

func TestValidatePatientProfile(t *testing.T) {
	tests := []struct {
		name             string
		patientID        string
		riskScoreCount   int
		clinicalCount    int
		wantErrCount     int
	}{
		{"valid profile", "p1", 2, 1, 0},
		{"empty patientID", "", 1, 1, 1},
		{"no risk scores", "p2", 0, 1, 1},
		{"no clinical features", "p3", 1, 0, 1},
		{"all missing", "", 0, 0, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidatePatientProfile(tt.patientID, tt.riskScoreCount, tt.clinicalCount)
			if len(errs) != tt.wantErrCount {
				t.Errorf("ValidatePatientProfile() errors = %v, wantErrCount %d", errs, tt.wantErrCount)
			}
		})
	}
}

func TestValidateRiskScore(t *testing.T) {
	tests := []struct {
		name      string
		locusID   string
		score     float64
		wantErrs  int
	}{
		{"valid rsID and score", "rs12345", 0.5, 0},
		{"invalid locus format", "rsabc", 0.5, 1},
		{"score below range", "rs1", -0.1, 1},
		{"score above range", "rs1", 1.1, 1},
		{"both invalid", "bad", 2.0, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidateRiskScore(tt.locusID, tt.score)
			if len(errs) != tt.wantErrs {
				t.Errorf("ValidateRiskScore() errors = %v, wantErrs %d", errs, tt.wantErrs)
			}
		})
	}
}
