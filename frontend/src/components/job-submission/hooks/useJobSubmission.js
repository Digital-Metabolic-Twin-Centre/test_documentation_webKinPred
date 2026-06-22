// /home/saleh/webKinPred/frontend/src/components/job-submission/hooks/useJobSubmission.js
import { useState, useMemo, useRef, useEffect } from 'react';
import {
  detectCsvFormat,
  fetchMethods,
  validateCsv,
  fetchSequenceSimilaritySummary,
  submitJob as submitJobApi,
  openProgressStream,
  cancelValidationApi
} from '../services/api';

const TARGET_ORDER = ['kcat', 'Km', 'kcat/Km'];
const makeEmptyMethods = () => ({ kcat: '', Km: '', 'kcat/Km': '' });

function makeSessionId() {
  return 'vs_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function useJobSubmission() {
  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [showPreprocessPrompt, setShowPreprocessPrompt] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [showValidationResults, setShowValidationResults] = useState(false);

  // Domain state
  const [selectedTargets, setSelectedTargets] = useState([]);
  const [targetMethods, setTargetMethods] = useState(makeEmptyMethods());
  const [csvFile, setCsvFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [useExperimental, setUseExperimental] = useState(false);
  const [includeSimilarityColumns, setIncludeSimilarityColumns] = useState(true);
  const [canonicalizeSubstrates, setCanonicalizeSubstrates] = useState(true);
  const [handleLongSeqs, setHandleLongSeqs] = useState('truncate');

  // Derived server feedback
  const [csvFormatInfo, setCsvFormatInfo] = useState(null);
  const [csvFormatValid, setCsvFormatValid] = useState(false);
  const [csvFormatError, setCsvFormatError] = useState('');
  const [csvParsing, setCsvParsing] = useState(false);
  const [similarityData, setSimilarityData] = useState(null);
  const [submissionResult, setSubmissionResult] = useState(null);

  // Live log state
  const [validationSessionId, setValidationSessionId] = useState('');
  const userCancelledRef = useRef(false);
  const [liveLogs, setLiveLogs] = useState([]);
  const [streamConnected, setStreamConnected] = useState(false);
  const [showLogOverlay, setShowLogOverlay] = useState(false);
  const [validationDone, setValidationDone] = useState(false);
  const eventSourceRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Method registry (fetched once from the backend)
  const [methods, setMethods] = useState(null);

  useEffect(() => {
    fetchMethods()
      .then(setMethods)
      .catch((err) => {
        console.error('Failed to fetch method registry from backend:', err);
      });
  }, []);

  const orderedTargets = useMemo(
    () => TARGET_ORDER.filter((target) => selectedTargets.includes(target)),
    [selectedTargets]
  );

  const selectedMethodsPayload = useMemo(() => {
    const out = {};
    for (const target of orderedTargets) {
      if (targetMethods[target]) out[target] = targetMethods[target];
    }
    return out;
  }, [orderedTargets, targetMethods]);

  const allSelectedTargetsHaveMethods = useMemo(
    () => orderedTargets.every((target) => Boolean(targetMethods[target])),
    [orderedTargets, targetMethods]
  );

  // Accepted schemas are target-specific: CatPred kcat consumes lists natively,
  // ordinary pair methods expand them, and TurNup requires Products.
  const allowedMethodsByTarget = useMemo(() => {
    const out = { kcat: [], Km: [], 'kcat/Km': [] };
    if (!methods || !csvFormatInfo?.csv_type) return out;

    const csvType = csvFormatInfo.csv_type;
    const canUseMethod = (methodMeta, target) => {
      const targetTypes = methodMeta.acceptedCsvTypesByTarget?.[target];
      if (Array.isArray(targetTypes)) {
        return targetTypes.includes(csvType);
      }
      if (Array.isArray(methodMeta.acceptedCsvTypes)) {
        return methodMeta.acceptedCsvTypes.includes(csvType);
      }
      if (csvType === 'full_reaction') return true;
      if (csvType === 'substrate_list') return methodMeta.inputFormat === 'single';
      return methodMeta.inputFormat === 'single';
    };

    for (const target of TARGET_ORDER) {
      out[target] = Object.entries(methods)
        .filter(([, meta]) => meta.supports.includes(target) && canUseMethod(meta, target))
        .map(([key]) => key);
    }
    return out;
  }, [csvFormatInfo, methods]);

  // Event handlers
  const resetMethods = () => {
    setTargetMethods(makeEmptyMethods());
  };

  const onChangeTargets = (nextTargets) => {
    const ordered = TARGET_ORDER.filter((target) => nextTargets.includes(target));
    setSelectedTargets(ordered);
    setTargetMethods((prev) => {
      const next = { ...prev };
      for (const target of TARGET_ORDER) {
        if (!ordered.includes(target)) next[target] = '';
      }
      return next;
    });
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);
  };

  const onMethodChange = (target, methodKey) => {
    setTargetMethods((prev) => ({ ...prev, [target]: methodKey }));
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);
  };

  const onFileSelected = async (file) => {
    setCsvFile(file);
    setFileName(file?.name || 'No file chosen');
    resetMethods();
    setSimilarityData(null);
    setSubmissionResult(null);
    setShowValidationResults(false);

    if (!file) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError('');
      return;
    }

    setCsvParsing(true);
    try {
      const data = await detectCsvFormat(file);
      if (data.status === 'valid') {
        setCsvFormatInfo(data);
        setCsvFormatValid(true);
        setCsvFormatError('');
      } else {
        setCsvFormatInfo(null);
        setCsvFormatValid(false);
        setCsvFormatError(
          Array.isArray(data.errors) ? data.errors.join('; ') : 'Invalid CSV format.'
        );
      }
    } catch (err) {
      setCsvFormatInfo(null);
      setCsvFormatValid(false);
      setCsvFormatError(err?.response?.data?.error || 'Error detecting CSV format.');
    } finally {
      setCsvParsing(false);
    }
  };

  const openStream = (sid) => {
    if (eventSourceRef.current) {
      try { eventSourceRef.current.close(); } catch { /* ignore */ }
    }
    const es = openProgressStream(sid);
    eventSourceRef.current = es;
    setLiveLogs([]);
    setStreamConnected(true);

    es.onmessage = (evt) => {
      if (!evt?.data) return;
      setLiveLogs((prev) => [...prev, evt.data]);
    };
    es.addEventListener('done', () => {
      es.close();
      eventSourceRef.current = null;
      setStreamConnected(false);
    });
    es.onerror = () => {
      setStreamConnected(false);
    };
  };

  const closeStream = () => {
    if (eventSourceRef.current) {
      try { eventSourceRef.current.close(); } catch { /* ignore */ }
      eventSourceRef.current = null;
    }
    setStreamConnected(false);
  };

  useEffect(() => {
    return () => closeStream();
  }, []);

  const closeLogOverlay = () => {
    setShowLogOverlay(false);
    setValidationDone(false);
    closeStream();
  };

  const runValidation = async (runSimilarity = true) => {
    if (!csvFile) return;
    const sid = makeSessionId();
    setValidationSessionId(sid);
    userCancelledRef.current = false;
    setValidationDone(false);
    setShowLogOverlay(true);
    openStream(sid);
    setIsValidating(true);

    try {
      const validation = await validateCsv({
        file: csvFile,
        targets: orderedTargets,
        methods: selectedMethodsPayload,
      });
      const {
        invalid_substrates,
        invalid_proteins,
        length_violations,
        length_limits,
      } = validation;

      if (runSimilarity) {
        const sim = await fetchSequenceSimilaritySummary({
          file: csvFile,
          useExperimental,
          validationSessionId: sid,
        });
        if (userCancelledRef.current) return;
        setSimilarityData(sim);
      } else {
        setSimilarityData(null);
      }

      setSubmissionResult({
        invalid_substrates,
        invalid_proteins,
        length_violations,
        length_limits,
      });
      setShowValidationResults(true);
    } catch (err) {
      if (!userCancelledRef.current) {
        alert('Validation failed. Please try again. ' + (err?.message || ''));
      }
    } finally {
      setIsValidating(false);
      setValidationDone(false);
      setShowLogOverlay(false);
      setTimeout(() => closeStream(), 5000);
    }
  };

  const cancelValidation = async () => {
    if (!validationSessionId) {
      setIsValidating(false);
      setShowLogOverlay(false);
      closeStream();
      return;
    }
    userCancelledRef.current = true;
    try {
      await cancelValidationApi(validationSessionId);
    } catch { /* swallow */ }
    setIsValidating(false);
    setValidationDone(false);
    setShowLogOverlay(false);
    closeStream();
  };

  const submitJob = async () => {
    if (!csvFile || orderedTargets.length === 0) return;
    if (!allSelectedTargetsHaveMethods) {
      alert('Please select one method for each chosen target before submitting.');
      return;
    }

    setIsSubmitting(true);
    try {
      const data = await submitJobApi({
        targets: orderedTargets,
        methods: selectedMethodsPayload,
        file: csvFile,
        handleLongSequences: handleLongSeqs,
        useExperimental,
        includeSimilarityColumns,
        canonicalizeSubstrates,
      });
      setSubmissionResult((prev) => ({
        ...prev,
        message: data.message,
        public_id: data.public_id,
      }));
      setShowModal(true);
    } catch (error) {
      const msg = error?.response?.data?.error || '';
      alert('Failed to submit job\n' + msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    // state
    isSubmitting,
    isValidating,
    showPreprocessPrompt,
    setShowPreprocessPrompt,
    showModal,
    setShowModal,
    showValidationResults,
    setShowValidationResults,

    selectedTargets,
    setSelectedTargets: onChangeTargets,
    targetMethods,
    setTargetMethod: onMethodChange,
    allSelectedTargetsHaveMethods,
    csvFile,
    fileName,
    csvFormatInfo,
    csvFormatValid,
    csvFormatError,
    csvParsing,
    useExperimental,
    setUseExperimental,
    includeSimilarityColumns,
    setIncludeSimilarityColumns,
    canonicalizeSubstrates,
    setCanonicalizeSubstrates,
    handleLongSeqs,
    setHandleLongSeqs,
    similarityData,
    submissionResult,

    // registry-derived method lists
    methods,
    allowedMethodsByTarget,

    // logs
    liveLogs,
    streamConnected,
    showLogOverlay,
    validationDone,
    autoScroll,
    setAutoScroll,

    // actions
    onFileSelected,
    runValidation,
    submitJob,
    cancelValidation,
    closeLogOverlay,
  };
}
