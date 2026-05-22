// /home/saleh/webKinPred/frontend/src/components/job-submission/JobSubmissionForm.jsx
import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';

import useJobSubmission from './hooks/useJobSubmission';
import HowToUseCard from './components/HowToUseCard';
import PredictionTypeSelect from './components/PredictionTypeSelect';
import CsvUpload from './components/CsvUpload';
import MethodPicker from './components/MethodPicker';
import ValidationResults from './components/ValidationResults';
import PreprocessModal from './components/PreprocessModal';
import SubmissionResultModal from './components/SubmissionResultModal';
import LiveLogOverlay from './components/LiveLogOverlay';

export default function JobSubmissionForm() {
  const state = useJobSubmission();

  return (
    <Container className="mt-1 pb-5">
      <Row className="justify-content-center">
        <Col md={10}>
          <HowToUseCard methods={state.methods || {}} />

          <PredictionTypeSelect
            value={state.selectedTargets}
            onChange={state.setSelectedTargets}
          />

          {state.selectedTargets.length > 0 && (
            <CsvUpload
              csvFormatValid={state.csvFormatValid}
              csvFormatInfo={state.csvFormatInfo}
              csvFormatError={state.csvFormatError}
              csvParsing={state.csvParsing}
              onFileSelected={state.onFileSelected}
              onClickValidate={() => state.setShowPreprocessPrompt(true)}
              fileName={state.fileName}
            />
          )}

          {state.submissionResult && (
            <ValidationResults
              submissionResult={state.submissionResult}
              showValidationResults={state.showValidationResults}
              setShowValidationResults={state.setShowValidationResults}
              handleLongSeqs={state.handleLongSeqs}
              setHandleLongSeqs={state.setHandleLongSeqs}
              similarityData={state.similarityData}
              methods={state.methods || {}}
            />
          )}

          {state.selectedTargets.length > 0 && state.csvFile && state.csvFormatValid && (
            <MethodPicker
              selectedTargets={state.selectedTargets}
              allowedMethodsByTarget={state.allowedMethodsByTarget}
              methods={state.methods}
              targetMethods={state.targetMethods}
              setTargetMethod={state.setTargetMethod}
              csvFormatInfo={state.csvFormatInfo}
              useExperimental={state.useExperimental}
              setUseExperimental={state.setUseExperimental}
              includeSimilarityColumns={state.includeSimilarityColumns}
              setIncludeSimilarityColumns={state.setIncludeSimilarityColumns}
              canonicalizeSubstrates={state.canonicalizeSubstrates}
              setCanonicalizeSubstrates={state.setCanonicalizeSubstrates}
              onSubmit={state.submitJob}
              isSubmitting={state.isSubmitting}
              allSelectedTargetsHaveMethods={state.allSelectedTargetsHaveMethods}
            />
          )}

          {/* Modals */}
          <PreprocessModal
            show={state.showPreprocessPrompt}
            onHide={() => state.setShowPreprocessPrompt(false)}
            onRunValidation={async (runSimilarity) => {
              state.setShowPreprocessPrompt(false);
              await state.runValidation(runSimilarity);
            }}
            isValidating={state.isValidating}
          />

          <SubmissionResultModal
            show={state.showModal}
            onHide={() => state.setShowModal(false)}
            message={state.submissionResult?.message}
            publicId={state.submissionResult?.public_id}
          />
        </Col>
      </Row>

      {/* New Live Progress Overlay */}
      <LiveLogOverlay
        show={state.showLogOverlay}
        done={state.validationDone}
        logs={state.liveLogs}
        connected={state.streamConnected}
        autoScroll={state.autoScroll}
        setAutoScroll={state.setAutoScroll}
        onCancel={state.cancelValidation}
        onClose={state.closeLogOverlay}
      />
    </Container>
  );
}
