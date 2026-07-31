package model

import (
	"encoding/json"
	"testing"
)

func TestCanonicalPatientProfileRoundTrip(t *testing.T) {
	profile := CanonicalPatientProfile{
		PatientID:          "p1",
		Sex:                "F",
		Ethnicity:          "EUR",
		AgeAtDiagnosisDays: 10950,
		RiskScores: []PolygenicRiskScoreEntry{
			{LocusID: "rs123", GeneSymbol: "HLA", ContinuousScore: 0.8, ZScore: 2.1, PValue: 0.001, SourceCohort: "GWAS"},
		},
		ClinicalFeatures: []ClinicalFeatureEntry{
			{FeatureID: "f1", FeatureName: "Thymoma", Value: true},
		},
		Annotations: map[string]string{"note": "synthetic"},
	}

	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded CanonicalPatientProfile
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.PatientID != profile.PatientID {
		t.Errorf("PatientID mismatch: got %s, want %s", decoded.PatientID, profile.PatientID)
	}
	if len(decoded.RiskScores) != 1 {
		t.Errorf("RiskScores length: got %d, want 1", len(decoded.RiskScores))
	}
	if decoded.Annotations["note"] != "synthetic" {
		t.Errorf("Annotations mismatch: got %s", decoded.Annotations["note"])
	}
}

func TestNormalizationResultJSON(t *testing.T) {
	result := NormalizationResult{
		BatchID:          "batch-1",
		ProfilesAccepted: 1,
		ProfilesRejected: 0,
		ValidationErrors: []string{},
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	var decoded NormalizationResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if decoded.BatchID != "batch-1" {
		t.Errorf("BatchID mismatch: got %s", decoded.BatchID)
	}
	if decoded.ProfilesAccepted != 1 {
		t.Errorf("ProfilesAccepted mismatch: got %d", decoded.ProfilesAccepted)
	}
}
