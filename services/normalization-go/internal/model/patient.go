package model

// CanonicalPatientProfile is the normalized internal representation
// of a patient after ingestion and cleaning.
type CanonicalPatientProfile struct {
	PatientID          string                    `json:"patient_id"`
	Sex                string                    `json:"sex"`
	Ethnicity          string                    `json:"ethnicity"`
	AgeAtDiagnosisDays uint32                    `json:"age_at_diagnosis_days"`
	RiskScores         []PolygenicRiskScoreEntry `json:"risk_scores"`
	ClinicalFeatures   []ClinicalFeatureEntry    `json:"clinical_features"`
	Annotations        map[string]string         `json:"annotations,omitempty"`
}

// PolygenicRiskScoreEntry represents a single locus risk score.
type PolygenicRiskScoreEntry struct {
	LocusID         string             `json:"locus_id"`
	GeneSymbol      string             `json:"gene_symbol"`
	ContinuousScore float32            `json:"continuous_score"`
	ZScore          float32            `json:"z_score"`
	PValue          float32            `json:"p_value"`
	SourceCohort    string             `json:"source_cohort"`
	Metadata        map[string]string  `json:"metadata,omitempty"`
}

// ClinicalFeatureEntry represents a single clinical feature.
type ClinicalFeatureEntry struct {
	FeatureID   string      `json:"feature_id"`
	FeatureName string      `json:"feature_name"`
	Value       interface{} `json:"value"`
}

// NormalizationResult holds the outcome of batch normalization.
type NormalizationResult struct {
	BatchID         string                  `json:"batch_id"`
	Profiles        []CanonicalPatientProfile `json:"profiles"`
	ValidationErrors []string               `json:"validation_errors"`
	ProfilesAccepted uint32                 `json:"profiles_accepted"`
	ProfilesRejected uint32                 `json:"profiles_rejected"`
}
