/**
 * CrimeVisionAI - Frontend API Service Module
 * Handles HTTP requests to the backend API.
 * Uses same-origin relative base URLs and adapts to existing endpoints.
 */

const CrimeVisionAPI = (function () {
  'use strict';

  const BASE_URL = '';

  /**
   * Send an audio file for speech transcription and anchor-event timeline extraction.
   * Calls real backend endpoint: POST /api/audio/analyze
   *
   * @param {File} file - Audio file (.wav, .mp3, .m4a, etc.)
   * @param {string|null} language - Optional language code ('en', 'te', 'hi', 'fr')
   * @returns {Promise<Object>} Consolidated transcription and timeline findings
   */
  async function analyzeAudio(file, language = null) {
    if (!file) {
      throw new Error('Please select an audio file to analyze.');
    }

    const formData = new FormData();
    formData.append('file', file);
    if (language) {
      formData.append('language', language);
    }

    const response = await fetch(`${BASE_URL}/api/audio/analyze`, {
      method: 'POST',
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      const errorMsg = data.message || data.error || `Request failed with status ${response.status}`;
      throw new Error(errorMsg);
    }

    return data;
  }

  /**
   * Batch evidence upload placeholder (Planned for Activity 6.1).
   */
  async function uploadBatch(files, caseId = null) {
    throw new Error(
      'Batch evidence upload endpoint is planned for Activity 6.1 and not yet exposed by the backend.'
    );
  }

  /**
   * Send an image file for forensic vision analysis.
   * Calls real backend endpoint: POST /api/image/analyze
   *
   * @param {File} file - Image file (.jpg, .jpeg, .png, .bmp, .webp, .tiff)
   * @param {number|null} confidenceThreshold - Optional confidence threshold (0.0 to 1.0)
   * @returns {Promise<Object>} Consolidated vision analysis findings
   */
  async function analyzeImage(file, confidenceThreshold = null) {
    if (!file) {
      throw new Error('Please select an image file to analyze.');
    }

    const formData = new FormData();
    formData.append('file', file);
    if (confidenceThreshold !== null && confidenceThreshold !== undefined) {
      formData.append('confidence_threshold', confidenceThreshold);
    }

    const response = await fetch(`${BASE_URL}/api/image/analyze`, {
      method: 'POST',
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      const errorMsg = data.message || data.error || `Request failed with status ${response.status}`;
      throw new Error(errorMsg);
    }

    return data;
  }

  /**
   * List all recorded investigation cases.
   * Calls: GET /api/cases
   * @returns {Promise<Array>} List of cases
   */
  async function listCases() {
    const response = await fetch(`${BASE_URL}/api/cases`);
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to list cases');
    }
    return data.cases || [];
  }

  /**
   * Create a new investigation case.
   * Calls: POST /api/cases
   * @param {string} title - Case title
   * @param {string} description - Case description
   * @param {string|null} caseId - Optional custom case ID
   * @returns {Promise<Object>} Created case object
   */
  async function createCase(title, description = '', caseId = null) {
    const payload = { title, description };
    if (caseId) {
      payload.case_id = caseId;
    }
    const response = await fetch(`${BASE_URL}/api/cases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to create case');
    }
    return data.case;
  }

  /**
   * Get case details, evidence list, and consolidated timeline.
   * Calls: GET /api/cases/<case_id>
   * @param {string} caseId - Target case ID
   * @returns {Promise<Object>} Case details
   */
  async function getCase(caseId) {
    if (!caseId) {
      throw new Error('Case ID is required.');
    }
    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}`);
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || `Failed to fetch case ${caseId}`);
    }
    return data.case;
  }

  /**
   * Upload an evidence file to a specific case and optionally analyze it.
   * Calls: POST /api/cases/<case_id>/evidence
   * @param {string} caseId - Target case ID
   * @param {File} file - Audio or image file
   * @param {Object} options - { mediaType, analyze, language, confidenceThreshold }
   * @returns {Promise<Object>} Upload and analysis result
   */
  async function uploadCaseEvidence(caseId, file, options = {}) {
    if (!caseId) {
      throw new Error('Case ID is required to upload evidence.');
    }
    if (!file) {
      throw new Error('Please select an evidence file to upload.');
    }
    const opts = typeof options === 'string' ? { mediaType: options } : (options || {});
    const formData = new FormData();
    formData.append('file', file);
    if (opts.mediaType) {
      formData.append('media_type', opts.mediaType);
    }
    formData.append('analyze', opts.analyze === false ? 'false' : 'true');
    if (opts.language) {
      formData.append('language', opts.language);
    }
    if (opts.confidenceThreshold !== undefined && opts.confidenceThreshold !== null) {
      formData.append('confidence_threshold', opts.confidenceThreshold);
    }

    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/evidence`, {
      method: 'POST',
      body: formData,
    });
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to upload case evidence');
    }
    if (data && data.evidence && data.evidence.evidence_id && !data.evidence_id) {
      data.evidence_id = data.evidence.evidence_id;
    }
    return data;
  }

  /**
   * Run or re-run analysis for existing case evidence.
   * Calls: POST /api/cases/<case_id>/evidence/<evidence_id>/analyze
   * @param {string} caseId
   * @param {string} evidenceId
   * @param {Object} options - { language, confidenceThreshold }
   * @returns {Promise<Object>}
   */
  async function analyzeCaseEvidence(caseId, evidenceId, options = {}) {
    const formData = new FormData();
    if (options.language) {
      formData.append('language', options.language);
    }
    if (options.confidenceThreshold !== undefined && options.confidenceThreshold !== null) {
      formData.append('confidence_threshold', options.confidenceThreshold);
    }

    const response = await fetch(
      `${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/evidence/${encodeURIComponent(evidenceId)}/analyze`,
      {
        method: 'POST',
        body: formData,
      }
    );
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to analyze evidence');
    }
    return data;
  }

  /**
   * Send a video file for sequential frame analysis and temporal event inference.
   * Calls real backend endpoint: POST /api/video/analyze
   *
   * @param {File} file - Video file (.mp4, .mov, .avi, .mkv, .webm)
   * @param {number|null} confidenceThreshold - Optional confidence threshold (0.0 to 1.0)
   * @returns {Promise<Object>} Consolidated video analysis findings
   */
  async function analyzeVideo(file, confidenceThreshold = null) {
    if (!file) {
      throw new Error('Please select a video file to analyze.');
    }

    const formData = new FormData();
    formData.append('file', file);
    if (confidenceThreshold !== null && confidenceThreshold !== undefined) {
      formData.append('confidence_threshold', confidenceThreshold);
    }

    const response = await fetch(`${BASE_URL}/api/video/analyze`, {
      method: 'POST',
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      const errorMsg = data.message || data.error || `Request failed with status ${response.status}`;
      throw new Error(errorMsg);
    }

    return data;
  }

  /**
   * Return safe streaming URL for an evidence record.
   */
  function getMediaUrl(evidenceId) {
    return `${BASE_URL}/api/evidence/${encodeURIComponent(evidenceId)}/media`;
  }

  /**
   * Retrieve structured cross-modal fusion results for a case.
   * Calls: GET /api/cases/<case_id>/fusion
   * @param {string} caseId
   * @returns {Promise<Object>}
   */
  async function getCaseFusion(caseId) {
    if (!caseId) {
      throw new Error('Case ID is required to retrieve fusion.');
    }
    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/fusion`);
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to retrieve case fusion');
    }
    return data.fusion;
  }

  /**
   * Execute or re-run multimodal fusion on case evidence.
   * Calls: POST /api/cases/<case_id>/fusion
   * @param {string} caseId
   * @param {Object|null} syncGroups
   * @returns {Promise<Object>}
   */
  async function fuseCase(caseId, syncGroups = null) {
    if (!caseId) {
      throw new Error('Case ID is required for cross-modal fusion.');
    }
    const payload = syncGroups ? { sync_groups: syncGroups } : {};
    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/fusion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Cross-modal fusion failed');
    }
    return data.fusion;
  }

  /**
   * Complete multimodal pipeline trigger executing cross-modal fusion (Activity 5.1).
   */
  async function processCompletePipeline(caseId) {
    return await fuseCase(caseId);
  }

  /**
   * Grounded Investigator Chat Q&A query against case findings.
   * Calls real backend endpoint: POST /api/cases/<case_id>/chat
   * Supports backward compatible fallback if invoked with (mode, sessionId, question).
   */
  async function chat(caseIdOrMode, questionOrSession, legacyQuestion = null) {
    // Check if called with (caseId, question) or legacy (mode, sessionId, question)
    let caseId = caseIdOrMode;
    let question = questionOrSession;
    if (legacyQuestion !== null && legacyQuestion !== undefined) {
      question = legacyQuestion;
    }

    if (!caseId) {
      throw new Error('Case ID is required for investigator chat.');
    }
    if (!question || !question.trim()) {
      throw new Error('Please enter a question to ask.');
    }

    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question.trim() }),
    });

    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }

    if (!response.ok) {
      throw new Error(data.message || data.error || 'Chat query failed');
    }

    return data;
  }

  /**
   * Retrieve structured incident hypothesis for a case.
   * Calls: GET /api/cases/<case_id>/hypothesis
   */
  async function getCaseHypothesis(caseId, refresh = false) {
    if (!caseId) {
      throw new Error('Case ID is required to retrieve hypothesis.');
    }
    const query = refresh ? '?refresh=true' : '';
    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/hypothesis${query}`);
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to retrieve incident hypothesis');
    }
    return data.hypothesis;
  }

  /**
   * Explicitly synthesize case evidence and generate evidence-grounded hypotheses.
   * Calls: POST /api/cases/<case_id>/hypothesis
   */
  async function evaluateCaseHypothesis(caseId) {
    if (!caseId) {
      throw new Error('Case ID is required for hypothesis evaluation.');
    }
    const response = await fetch(`${BASE_URL}/api/cases/${encodeURIComponent(caseId)}/hypothesis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    let data;
    try {
      data = await response.json();
    } catch (e) {
      throw new Error(`Server returned non-JSON response (HTTP ${response.status})`);
    }
    if (!response.ok) {
      throw new Error(data.message || data.error || 'Hypothesis evaluation failed');
    }
    return data.hypothesis;
  }

  return {
    analyzeAudio,
    uploadBatch,
    analyzeImage,
    analyzeVideo,
    getMediaUrl,
    listCases,
    createCase,
    getCase,
    uploadCaseEvidence,
    uploadEvidence: uploadCaseEvidence,
    analyzeCaseEvidence,
    getCaseFusion,
    fuseCase,
    getCaseHypothesis,
    evaluateCaseHypothesis,
    processCompletePipeline,
    chat,
  };
})();

// Export globally for browser scripts
window.CrimeVisionAPI = CrimeVisionAPI;
