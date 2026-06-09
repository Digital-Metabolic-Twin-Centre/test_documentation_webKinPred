// /home/saleh/webKinPred/frontend/src/components/job-submission/services/api.js
import apiClient from '../../appClient';

/**
 * Fetch the full method registry from the backend.
 *
 * The API returns methods grouped by target:
 * { kcat: [{id, ...}], Km: [{id, ...}], "kcat/Km": [{id, ...}] }.
 * This function flattens that into a plain object keyed by method id, e.g.:
 *   {
 *     "DLKcat":  { id: "DLKcat", displayName, supports: ["kcat"], ... },
 *     "UniKP":   { id: "UniKP", supports: ["kcat", "Km"], ... },
 *     "RealKcat": { id: "RealKcat", supports: ["kcat", "Km"], ... },
 *     "IECata":   { id: "IECata", supports: ["kcat/Km"], ... },
 *     "CataPro": { id: "CataPro", supports: ["kcat", "Km", "kcat/Km"], ... },
 *     ...
 *   }
 *
 * No authentication required.
 */
export async function fetchMethods() {
  const { data } = await apiClient.get('/v1/methods/');
  const flat = {};
  for (const group of Object.values(data.methods)) {
    for (const m of group) {
      flat[m.id] = m;
    }
  }
  return flat;
}

export async function detectCsvFormat(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await apiClient.post('/detect-csv-format/', formData);
  return data;
}

export async function validateCsv({ file, targets, methods }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('targets', JSON.stringify(targets || []));
  formData.append('methods', JSON.stringify(methods || {}));
  const { data } = await apiClient.post('/validate-input/', formData);
  return data;
}

export async function fetchSequenceSimilaritySummary({ file, useExperimental, validationSessionId }) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('useExperimental', useExperimental);
  if (validationSessionId) formData.append('validationSessionId', validationSessionId);
  const { data } = await apiClient.post('/sequence-similarity-summary/', formData);
  if (data && data.error) {
    throw new Error(data.error);
  }
  return data;
}

export async function submitJob({
  targets,
  methods,
  file,
  handleLongSequences,
  useExperimental,
  includeSimilarityColumns,
  canonicalizeSubstrates,
}) {
  const formData = new FormData();
  formData.append('targets', JSON.stringify(targets || []));
  formData.append('methods', JSON.stringify(methods || {}));
  formData.append('file', file);
  formData.append('handleLongSequences', handleLongSequences);
  formData.append('useExperimental', useExperimental);
  formData.append('includeSimilarityColumns', includeSimilarityColumns);
  formData.append('canonicalizeSubstrates', canonicalizeSubstrates);
  const { data } = await apiClient.post('/submit-job/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}
export function openProgressStream(sessionId) {
  const baseURL = import.meta.env.VITE_API_BASE_URL || '/api/';
  const url = `${baseURL}/progress-stream/?session_id=${encodeURIComponent(sessionId)}`;
  return new EventSource(url);
}
export async function cancelValidationApi(sessionId) {
  const formData = new FormData();
  formData.append('session_id', sessionId);
  const { data } = await apiClient.post('/cancel-validation/', formData);
  return data;
}
