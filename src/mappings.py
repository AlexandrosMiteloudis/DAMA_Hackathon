"""
This module defines mappings and constants used across the project.
"""

KNOWN_CLINICAL_FEATURES: frozenset[str] = frozenset({
    'age_at_diagnosis',
    'chemotherapy',
    'cohort',
    'hormone_therapy',
    'lymph_nodes_examined_positive',
    'mutation_count',
    'neoplasm_histologic_grade',
    'nottingham_prognostic_index',
    'radio_therapy',
    'tumor_size',
    'tumor_stage',
})