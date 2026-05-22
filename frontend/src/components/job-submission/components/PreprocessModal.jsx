import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Modal, Button, Form } from 'react-bootstrap';

export default function PreprocessModal({
  show,
  onHide,
  onRunValidation,
  isValidating,
}) {
  const [runSimilarity, setRunSimilarity] = useState(true);

  // Reset to default-on whenever the modal opens
  useEffect(() => {
    if (show) setRunSimilarity(true);
  }, [show]);

  return (
    <Modal show={show} onHide={onHide} size="xl">
      <Modal.Header closeButton>
        <Modal.Title>Preprocess Before Prediction?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <p>Would you like to validate your input data before running predictions?</p>
        <p>
          This will identify invalid SMILES/InChIs and protein sequences and flag
          any rows that exceed per-method sequence length limits.
        </p>
        <p>
          <strong>Note:</strong> Even if you skip this step, invalid rows will be
          automatically excluded during prediction and will not produce results.
        </p>
        <p className="fw-bold">Recommended if you&apos;re unsure about input quality.</p>

        <hr className="my-3 opacity-25" />

        <div className="d-flex align-items-start gap-3">
          <Form.Check
            type="switch"
            id="run-similarity-toggle"
            checked={runSimilarity}
            onChange={(e) => setRunSimilarity(e.target.checked)}
            className="mt-1 flex-shrink-0"
          />
          <div>
            <label
              htmlFor="run-similarity-toggle"
              className="fw-semibold"
              style={{ cursor: 'pointer' }}
            >
              Include sequence similarity analysis
            </label>
            <p className="mb-0 small text-white-50 mt-1">
              Compares your input sequences against the training datasets of each
              prediction method using MMseqs2. Adds a similarity histogram to the
              results. Takes additional time depending on input size.
            </p>
          </div>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button
          className="btn kave-btn-run-val"
          onClick={onHide}
          disabled={isValidating}
        >
          Cancel
        </Button>
        <Button
          className="btn kave-btn-run-val"
          onClick={() => onRunValidation(runSimilarity)}
          disabled={isValidating}
        >
          {isValidating ? 'Validating…' : 'Run Validation'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

PreprocessModal.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  onRunValidation: PropTypes.func.isRequired,
  isValidating: PropTypes.bool.isRequired,
};
