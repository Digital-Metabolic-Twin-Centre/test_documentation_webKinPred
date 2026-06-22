import React from 'react';
import PropTypes from 'prop-types';
import { Card, Row, Col, Form, Button } from 'react-bootstrap';
import MethodDetails from './MethodDetails';
import ExperimentalSwitch from './ExperimentalSwitch';
import CanonicalizationSwitch from './CanonicalizationSwitch';
import SimilarityColumnsSwitch from './SimilarityColumnsSwitch';
import '../../../styles/components/PredictionTypeSelect.css';

const TARGET_ORDER = ['kcat', 'Km', 'kcat/Km'];

const TARGET_LABELS = {
  kcat: 'kcat',
  Km: 'KM',
  'kcat/Km': 'kcat/Km',
};

export default function MethodPicker({
  selectedTargets,
  allowedMethodsByTarget,
  methods,
  targetMethods,
  setTargetMethod,
  csvFormatInfo,
  useExperimental,
  setUseExperimental,
  includeSimilarityColumns,
  setIncludeSimilarityColumns,
  canonicalizeSubstrates,
  setCanonicalizeSubstrates,
  onSubmit,
  isSubmitting,
  allSelectedTargetsHaveMethods,
}) {
  const visibleTargets = TARGET_ORDER.filter((target) => selectedTargets.includes(target));

  const methodLabel = (key) => methods?.[key]?.displayName ?? key;
  const isListFormat = ['substrate_list', 'full_reaction'].includes(csvFormatInfo?.csv_type);

  const methodBehavior = (key, target) => (
    methods?.[key]?.inputBehaviorByTarget?.[target] || 'expanded_pair'
  );

  const optionLabel = (key, target) => {
    const label = methodLabel(key);
    if (!isListFormat) return label;
    const behavior = methodBehavior(key, target);
    if (behavior === 'native_full_reaction') return `${label} — native full reaction`;
    if (behavior === 'native_multi') return `${label} — native multi-substrate`;
    return target === 'kcat'
      ? `${label} — per-substrate maximum`
      : `${label} — per-substrate array`;
  };

  const groupedMethods = (target) => {
    const keys = allowedMethodsByTarget[target] || [];
    if (!isListFormat) return { native: [], expanded: keys };
    return {
      native: keys.filter((key) => methodBehavior(key, target) !== 'expanded_pair'),
      expanded: keys.filter((key) => methodBehavior(key, target) === 'expanded_pair'),
    };
  };

  return (
    <Card className="section-container section-method-selection mb-4">
      <Card.Header as="h3" className="text-center">
        Select Prediction Method(s)
      </Card.Header>
      <Card.Body>
        <Row>
          {visibleTargets.map((target) => (
            <Col key={target} md={visibleTargets.length > 1 ? 6 : 12} className="mb-3">
              <Form.Group controlId={`method-${target.replace('/', '-')}`} className="method-picker-group">
                <Form.Label className="method-picker-label">
                  Method for {TARGET_LABELS[target]}
                </Form.Label>
                <div className={`kave-select-wrapper ${targetMethods[target] ? 'is-selected' : ''}`}>
                  <Form.Select
                    disabled={!csvFormatInfo?.csv_type}
                    value={targetMethods[target] || ''}
                    onChange={(e) => setTargetMethod(target, e.target.value)}
                    className="kave-select"
                    required
                    aria-label={`Method for ${TARGET_LABELS[target]}`}
                  >
                    <option value="">Select method...</option>
                    {groupedMethods(target).native.length > 0 && (
                      <optgroup label="Native multi/full-reaction methods">
                        {groupedMethods(target).native.map((key) => (
                          <option key={key} value={key}>{optionLabel(key, target)}</option>
                        ))}
                      </optgroup>
                    )}
                    {groupedMethods(target).expanded.length > 0 && (
                      <optgroup label={isListFormat ? 'Single-substrate methods applied per substrate' : 'Methods'}>
                        {groupedMethods(target).expanded.map((key) => (
                          <option key={key} value={key}>{optionLabel(key, target)}</option>
                        ))}
                      </optgroup>
                    )}
                  </Form.Select>
                </div>
              </Form.Group>
              {targetMethods[target] && (
                <MethodDetails methodKey={targetMethods[target]} methods={methods} citationOnly />
              )}
            </Col>
          ))}
        </Row>
      </Card.Body>

      {visibleTargets.length > 0 && (
        <Card.Footer className="d-flex justify-content-end align-items-center gap-3 flex-wrap">
          <CanonicalizationSwitch
            checked={canonicalizeSubstrates}
            onChange={setCanonicalizeSubstrates}
          />
          <ExperimentalSwitch checked={useExperimental} onChange={setUseExperimental} />
          <SimilarityColumnsSwitch
            checked={includeSimilarityColumns}
            onChange={setIncludeSimilarityColumns}
          />
          <Button
            className="kave-btn ms-3"
            onClick={onSubmit}
            disabled={isSubmitting || !allSelectedTargetsHaveMethods}
          >
            {isSubmitting ? 'Submitting…' : 'Submit Job'}
          </Button>
        </Card.Footer>
      )}
    </Card>
  );
}

MethodPicker.propTypes = {
  selectedTargets: PropTypes.arrayOf(PropTypes.string).isRequired,
  allowedMethodsByTarget: PropTypes.object.isRequired,
  methods: PropTypes.object,
  targetMethods: PropTypes.object.isRequired,
  setTargetMethod: PropTypes.func.isRequired,
  csvFormatInfo: PropTypes.object,
  useExperimental: PropTypes.bool.isRequired,
  setUseExperimental: PropTypes.func.isRequired,
  includeSimilarityColumns: PropTypes.bool.isRequired,
  setIncludeSimilarityColumns: PropTypes.func.isRequired,
  canonicalizeSubstrates: PropTypes.bool.isRequired,
  setCanonicalizeSubstrates: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  isSubmitting: PropTypes.bool.isRequired,
  allSelectedTargetsHaveMethods: PropTypes.bool.isRequired,
};
