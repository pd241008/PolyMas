package schema

import (
	"fmt"
	"regexp"
)

var validLocusID = regexp.MustCompile(`^rs\d+$`)

// ValidatePatientProfile runs canonical validation rules on a normalized profile.
func ValidatePatientProfile(patientID string, riskScoreCount, clinicalFeatureCount int) []string {
	var errs []string

	if patientID == "" {
		errs = append(errs, "patient_id is required")
	}

	if riskScoreCount == 0 {
		errs = append(errs, "at least one risk score is required")
	}

	if clinicalFeatureCount == 0 {
		errs = append(errs, "at least one clinical feature is required")
	}

	return errs
}

// ValidateRiskScore validates a single risk score entry.
func ValidateRiskScore(locusID string, score float64) []string {
	var errs []string

	if !validLocusID.MatchString(locusID) {
		errs = append(errs, fmt.Sprintf("invalid locus_id format: %s (expected rs[0-9]+)", locusID))
	}

	if score < 0 || score > 1 {
		errs = append(errs, fmt.Sprintf("continuous_score out of range: %f (expected 0.0-1.0)", score))
	}

	return errs
}
