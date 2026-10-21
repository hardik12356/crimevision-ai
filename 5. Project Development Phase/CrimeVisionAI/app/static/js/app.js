/**
 * CrimeVisionAI — Obsidian Forensics Application Controller
 * Manages forensic case investigations, evidence lockers, primary media inspection stage
 * (Video player, Image viewer canvas, Audio waveforms), timeline seeking, multimodal fusion,
 * grounded investigator Q&A, and theme switching.
 */

(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // Shared Evidence State Model (Session + SQLite synced)
  // -------------------------------------------------------------------------
  const AppState = {
    // Active Case
    currentCaseId: 'CASE-2026-MVP-001',
    currentCase: null,
    cases: [],

    // Currently inspected evidence item in center stage
    selectedEvidenceId: null,

    // Multimodal Fusion (Activity 5.1 & Epic 6.1)
    fusion: null,

    // Case Incident Hypothesis
    caseHypothesis: null,

    // Audio viewer state
    status: 'idle', // 'idle' | 'processing' | 'completed' | 'error'
    evidence: null,
    transcription: null,
    timeline: null,
    events: [],
    error: null,

    // Image viewer state
    imageStatus: 'idle',
    imageEvidence: null,
    imageDetections: [],
    imageSceneResult: null,
    imageCaptionResult: null,
    imageLlmAnalysis: null,
    imageObjectDetection: null,
    imageError: null,

    // Video viewer state (Epic 4)
    videoStatus: 'idle',
    videoEvidence: null,
    videoResult: null,
    videoObservations: [],
    videoEvents: [],
    videoHypothesis: null,
    videoError: null,
    videoPreviewUrl: null,

    // Active Ingestion Tab (Decoupled from evidence inspection)
    activeTab: 'audio', // 'audio' | 'image' | 'video'

    resetAudio() {
      this.status = 'idle';
      this.evidence = null;
      this.transcription = null;
      this.timeline = null;
      this.events = [];
      this.error = null;
    },

    resetImage() {
      this.imageStatus = 'idle';
      this.imageEvidence = null;
      this.imageDetections = [];
      this.imageSceneResult = null;
      this.imageCaptionResult = null;
      this.imageLlmAnalysis = null;
      this.imageObjectDetection = null;
      this.imageError = null;
    },

    resetVideo() {
      this.videoStatus = 'idle';
      this.videoEvidence = null;
      this.videoResult = null;
      this.videoObservations = [];
      this.videoEvents = [];
      this.videoHypothesis = null;
      this.videoError = null;
      if (this.videoPreviewUrl && this.videoPreviewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(this.videoPreviewUrl);
      }
      this.videoPreviewUrl = null;
    },
  };

  // -------------------------------------------------------------------------
  // DOM Elements Cache
  // -------------------------------------------------------------------------
  const elements = {
    // Header
    headerCaseBadge: document.getElementById('header-case-badge'),
    headerAudioBadge: document.getElementById('header-audio-badge'),
    headerImageBadge: document.getElementById('header-image-badge'),
    headerVideoBadge: document.getElementById('header-video-badge'),
    btnThemeToggle: document.getElementById('btn-theme-toggle'),

    // Case Bar
    caseSelect: document.getElementById('case-select'),
    btnOpenNewCase: document.getElementById('btn-open-new-case'),
    caseTitleDisplay: document.getElementById('case-title-display'),
    caseDescDisplay: document.getElementById('case-desc-display'),

    // Modal
    newCaseModal: document.getElementById('new-case-modal'),
    modalAlertBox: document.getElementById('modal-alert-box'),
    inputNewCaseTitle: document.getElementById('new-case-title'),
    inputNewCaseId: document.getElementById('new-case-id'),
    inputNewCaseDesc: document.getElementById('new-case-desc'),
    btnSubmitNewCase: document.getElementById('btn-submit-new-case'),
    btnCancelNewCase: document.getElementById('btn-cancel-new-case'),
    btnCloseModal: document.getElementById('btn-close-modal'),

    // Stats
    statEvidenceCount: document.getElementById('stat-evidence-count'),
    statEventCount: document.getElementById('stat-event-count'),
    statDuration: document.getElementById('stat-duration'),
    statDurationLabel: document.getElementById('stat-duration-label'),
    statDurationSub: document.getElementById('stat-duration-sub'),
    statStatus: document.getElementById('stat-status'),

    // Case Evidence Locker
    caseEvidencePanel: document.getElementById('case-evidence-panel'),
    caseEvidenceCountBadge: document.getElementById('case-evidence-count-badge'),
    evidenceEmptyNotice: document.getElementById('evidence-empty-notice'),
    caseEvidenceList: document.getElementById('case-evidence-list'),

    // Ingestion Tabs & Panels
    tabAudio: document.getElementById('tab-audio'),
    tabImage: document.getElementById('tab-image'),
    tabVideo: document.getElementById('tab-video'),
    audioUploadSection: document.getElementById('audio-upload-section'),
    imageUploadSection: document.getElementById('image-upload-section'),
    videoUploadSection: document.getElementById('video-upload-section'),
    audioFactsPanel: document.getElementById('audio-facts-panel'),
    imageResultsPanel: document.getElementById('image-results-panel'),
    videoResultsPanel: document.getElementById('video-results-panel'),
    ingestionPanelIcon: document.getElementById('ingestion-panel-icon'),
    alertBox: document.getElementById('upload-alert-box'),

    // Center Media Stage Canvas Elements
    mediaStageCanvas: document.getElementById('media-stage-canvas'),
    imageViewerContainer: document.getElementById('image-viewer-container'),
    imagePreview: document.getElementById('image-preview'),
    audioViewerContainer: document.getElementById('audio-viewer-container'),
    mediaEmptyPlaceholder: document.getElementById('media-empty-placeholder'),
    inspectingFilename: document.getElementById('inspecting-filename'),
    inspectingBadge: document.getElementById('inspecting-badge'),
    inspectingMediaTypeIcon: document.getElementById('inspecting-media-type-icon'),

    // Incident Hypothesis Section (Directly below Primary Inspection Canvas)
    incidentHypothesisPanel: document.getElementById('incident-hypothesis-panel'),
    hypothesisStatusBadge: document.getElementById('hypothesis-status-badge'),
    btnRefreshHypothesis: document.getElementById('btn-refresh-hypothesis'),
    btnRetryHypothesis: document.getElementById('btn-retry-hypothesis'),
    hypothesisLoading: document.getElementById('hypothesis-loading'),
    hypothesisEmpty: document.getElementById('hypothesis-empty'),
    hypothesisEmptyTitle: document.getElementById('hypothesis-empty-title'),
    hypothesisEmptySub: document.getElementById('hypothesis-empty-sub'),
    hypothesisError: document.getElementById('hypothesis-error'),
    hypothesisErrorSub: document.getElementById('hypothesis-error-sub'),
    hypothesisContent: document.getElementById('hypothesis-content'),
    hypPrimaryType: document.getElementById('hyp-primary-type'),
    hypPlausibilityBadge: document.getElementById('hyp-plausibility-badge'),
    hypPrimaryNarrative: document.getElementById('hyp-primary-narrative'),
    hypAlternativesGroup: document.getElementById('hyp-alternatives-group'),
    hypAlternativesList: document.getElementById('hyp-alternatives-list'),
    hypSupportingGroup: document.getElementById('hyp-supporting-group'),
    hypSupportingList: document.getElementById('hyp-supporting-list'),
    hypContradictionsGroup: document.getElementById('hyp-contradictions-group'),
    hypContradictionsList: document.getElementById('hyp-contradictions-list'),
    hypGapsGroup: document.getElementById('hyp-gaps-group'),
    hypGapsList: document.getElementById('hyp-gaps-list'),
    hypAssessmentGroup: document.getElementById('hyp-assessment-group'),
    hypOverallAssessment: document.getElementById('hyp-overall-assessment'),

    // Audio Ingestion
    dropzone: document.getElementById('audio-dropzone'),
    fileInput: document.getElementById('audio-file-input'),
    selectedFileBox: document.getElementById('selected-file-box'),
    selectedFileName: document.getElementById('selected-file-name'),
    languageSelect: document.getElementById('language-select'),
    btnAnalyze: document.getElementById('btn-analyze-audio'),
    btnReset: document.getElementById('btn-reset-audio'),

    // Audio Facts
    factsContainer: document.getElementById('facts-container'),
    factFile: document.getElementById('fact-file'),
    factDuration: document.getElementById('fact-duration'),
    factLanguage: document.getElementById('fact-language'),
    factQuality: document.getElementById('fact-quality'),
    transcriptDisplay: document.getElementById('transcript-display'),
    segmentsList: document.getElementById('segments-list'),

    // Image Ingestion
    imageDropzone: document.getElementById('image-dropzone'),
    imageFileInput: document.getElementById('image-file-input'),
    imageSelectedFileBox: document.getElementById('image-selected-file-box'),
    imageSelectedFileName: document.getElementById('image-selected-file-name'),
    confidenceThreshold: document.getElementById('confidence-threshold'),
    btnAnalyzeImage: document.getElementById('btn-analyze-image'),
    btnResetImage: document.getElementById('btn-reset-image'),

    // Image Results
    imageFactsContainer: document.getElementById('image-facts-container'),
    imageFactFile: document.getElementById('image-fact-file'),
    imageFactScene: document.getElementById('image-fact-scene'),
    imageFactPersons: document.getElementById('image-fact-persons'),
    imageDetectionCountBadge: document.getElementById('image-detection-count-badge'),
    imageCaptionGroup: document.getElementById('image-caption-group'),
    imageCaptionDisplay: document.getElementById('image-caption-display'),
    imageDetectionsGroup: document.getElementById('image-detections-group'),
    imageCountNote: document.getElementById('image-count-note'),
    imageDetectionsList: document.getElementById('image-detections-list'),
    imageSceneGroup: document.getElementById('image-scene-group'),
    imageSceneList: document.getElementById('image-scene-list'),
    imageLlmGroup: document.getElementById('image-llm-group'),
    imageLlmDisplay: document.getElementById('image-llm-display'),

    // Video Ingestion (Epic 4)
    videoDropzone: document.getElementById('video-dropzone'),
    videoFileInput: document.getElementById('video-file-input'),
    videoSelectedFileBox: document.getElementById('video-selected-file-box'),
    videoSelectedFileName: document.getElementById('video-selected-file-name'),
    videoConfidenceThreshold: document.getElementById('video-confidence-threshold'),
    btnAnalyzeVideo: document.getElementById('btn-analyze-video'),
    btnResetVideo: document.getElementById('btn-reset-video'),

    // Video Results (Epic 4)
    videoPlayerPreview: document.getElementById('video-player-preview'),
    videoFramesCountBadge: document.getElementById('video-frames-count-badge'),
    videoFactFile: document.getElementById('video-fact-file'),
    videoFactDuration: document.getElementById('video-fact-duration'),
    videoFactSpan: document.getElementById('video-fact-span'),
    videoFactStatus: document.getElementById('video-fact-status'),
    videoHypType: document.getElementById('video-hyp-type'),
    videoHypDesc: document.getElementById('video-hyp-desc'),
    videoEventsList: document.getElementById('video-events-list'),
    videoFramesList: document.getElementById('video-frames-list'),

    // Timeline
    timelineEmptyNotice: document.getElementById('timeline-empty-notice'),
    timelineEventsList: document.getElementById('timeline-events-list'),
    timelineCountBadge: document.getElementById('timeline-count-badge'),

    // Multimodal Fusion (Epic 6.1 & Investigator Synthesis)
    btnRunFusion: document.getElementById('btn-run-fusion'),
    btnReFuse: document.getElementById('btn-re-fuse'),
    fusionMainContainer: document.getElementById('fusion-main-container'),
    fusionSummaryCard: document.getElementById('fusion-summary-card'),
    fusionAlignmentBadge: document.getElementById('fusion-alignment-badge'),
    fusionAnchorsBadge: document.getElementById('fusion-anchors-badge'),
    fusionMetricsGrid: document.getElementById('fusion-metrics-grid'),
    fusionMetricRelations: document.getElementById('fusion-metric-relations'),
    fusionMetricConflicts: document.getElementById('fusion-metric-conflicts'),
    fusionMetricReview: document.getElementById('fusion-metric-review'),
    fusionMetricEvents: document.getElementById('fusion-metric-events'),
    fusionSynthesisText: document.getElementById('fusion-synthesis-text'),
    fusionGapsBanner: document.getElementById('fusion-gaps-banner'),
    fusionGapsContent: document.getElementById('fusion-gaps-content'),
    fusionKeyFindingsCard: document.getElementById('fusion-key-findings-card'),
    fusionFindingsCountBadge: document.getElementById('fusion-findings-count-badge'),
    fusionKeyFindingsList: document.getElementById('fusion-key-findings-list'),
    fusionRelationshipsCard: document.getElementById('fusion-relationships-card'),
    fusionRelCountBadge: document.getElementById('fusion-rel-count-badge'),
    fusionRelationshipsList: document.getElementById('fusion-relationships-list'),
    fusionAuditDetails: document.getElementById('fusion-audit-details'),
    fusionNarrativeText: document.getElementById('fusion-narrative-text'),
    fusionHashMeta: document.getElementById('fusion-hash-meta'),
    fusionEventsTableWrap: document.getElementById('fusion-events-table-wrap'),
    fusionEmptyNotice: document.getElementById('fusion-empty-notice'),

    // Export & Grounded Chat
    btnExportJson: document.getElementById('btn-export-json'),
    btnExportPdf: document.getElementById('btn-export-pdf'),
    exportStatus: document.getElementById('export-status'),
    chatInput: document.getElementById('chat-input'),
    btnChatSend: document.getElementById('btn-chat-send'),
    chatMessagesContainer: document.getElementById('chat-messages-container'),

    // Right Pane Investigation Tabs
    invTabTimeline: document.getElementById('inv-tab-timeline'),
    invTabFusion: document.getElementById('inv-tab-fusion'),
    invTabChat: document.getElementById('inv-tab-chat'),
    invTabAll: document.getElementById('inv-tab-all'),
    timelineSectionWrap: document.getElementById('timeline-section-wrap'),
    fusionSectionWrap: document.getElementById('fusion-section-wrap'),
    chatSectionWrap: document.getElementById('chat-section-wrap'),

    // Footer
    footerCaseId: document.getElementById('footer-case-id'),
  };

  let selectedFile = null;
  let selectedImageFile = null;
  let selectedVideoFile = null;

  // -------------------------------------------------------------------------
  // Initialization & Event Binding
  // -------------------------------------------------------------------------
  async function init() {
    initTheme();
    bindInvestigationNavEvents();
    bindChatQueryChips();
    bindCaseEvents();
    bindFusionEvents();
    bindHypothesisEvents();
    bindUploadEvents();
    bindImageUploadEvents();
    bindVideoUploadEvents();
    bindTabEvents();
    bindExportEvents();
    bindChatEvents();

    await loadCasesList();
  }

  // -------------------------------------------------------------------------
  // Theme Switching & Persistence
  // -------------------------------------------------------------------------
  function initTheme() {
    let savedTheme = 'dark';
    try {
      savedTheme = localStorage.getItem('crimevision_theme') || 'dark';
    } catch (_) {}
    applyTheme(savedTheme);

    if (elements.btnThemeToggle) {
      elements.btnThemeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        try {
          localStorage.setItem('crimevision_theme', next);
        } catch (_) {}
      });
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);

    if (elements.btnThemeToggle) {
      const isDark = theme === 'dark';
      elements.btnThemeToggle.innerHTML = isDark
        ? '<span class="theme-icon" aria-hidden="true">☀️</span><span class="theme-label">Light Mode</span>'
        : '<span class="theme-icon" aria-hidden="true">🌙</span><span class="theme-label">Dark Mode</span>';
      elements.btnThemeToggle.setAttribute('aria-label', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
      elements.btnThemeToggle.setAttribute('title', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
  }

  // -------------------------------------------------------------------------
  // Right Pane Sub-Navigation (Timeline / Fusion / Ask AI)
  // -------------------------------------------------------------------------
  function bindInvestigationNavEvents() {
    const tabs = [
      { btn: elements.invTabTimeline, section: elements.timelineSectionWrap },
      { btn: elements.invTabFusion, section: elements.fusionSectionWrap },
      { btn: elements.invTabChat, section: elements.chatSectionWrap },
    ];

    function setInvTab(activeId) {
      if (activeId === 'all') {
        [elements.invTabTimeline, elements.invTabFusion, elements.invTabChat].forEach((b) => {
          if (b) b.classList.remove('active-inv-tab');
        });
        if (elements.invTabAll) elements.invTabAll.classList.add('active-inv-tab');
        tabs.forEach((t) => {
          if (t.section) t.section.style.display = 'block';
        });
        return;
      }

      if (elements.invTabAll) elements.invTabAll.classList.remove('active-inv-tab');
      tabs.forEach((t) => {
        if (!t.btn || !t.section) return;
        if (t.btn.id === activeId) {
          t.btn.classList.add('active-inv-tab');
          t.btn.setAttribute('aria-selected', 'true');
          t.section.style.display = 'block';
        } else {
          t.btn.classList.remove('active-inv-tab');
          t.btn.setAttribute('aria-selected', 'false');
          t.section.style.display = 'none';
        }
      });
    }

    if (elements.invTabTimeline) elements.invTabTimeline.addEventListener('click', () => setInvTab('inv-tab-timeline'));
    if (elements.invTabFusion) elements.invTabFusion.addEventListener('click', () => setInvTab('inv-tab-fusion'));
    if (elements.invTabChat) elements.invTabChat.addEventListener('click', () => setInvTab('inv-tab-chat'));
    if (elements.invTabAll) elements.invTabAll.addEventListener('click', () => setInvTab('all'));
  }

  function bindChatQueryChips() {
    const chips = document.querySelectorAll('.chat-query-chips .chip-btn');
    chips.forEach((btn) => {
      btn.addEventListener('click', () => {
        const query = btn.dataset.query;
        if (query && elements.chatInput) {
          elements.chatInput.value = query;
          handleChatSubmit();
        }
      });
    });
  }

  // -------------------------------------------------------------------------
  // Case Management Logic
  // -------------------------------------------------------------------------
  async function loadCasesList() {
    try {
      const cases = await CrimeVisionAPI.listCases();
      AppState.cases = cases;
      populateCaseDropdown(cases);

      const targetId = AppState.currentCaseId || (cases.length > 0 ? cases[0].case_id : null);
      if (targetId) {
        await switchCase(targetId);
      }
    } catch (err) {
      showAlert(`Unable to load cases: ${err.message}`, 'error');
    }
  }

  function populateCaseDropdown(cases) {
    if (!elements.caseSelect) return;
    while (elements.caseSelect.firstChild) {
      elements.caseSelect.removeChild(elements.caseSelect.firstChild);
    }

    cases.forEach((c) => {
      const opt = document.createElement('option');
      opt.value = c.case_id;
      opt.textContent = `${c.case_id} — ${c.title}`;
      elements.caseSelect.appendChild(opt);
    });
  }

  async function switchCase(caseId) {
    if (!caseId) return;
    try {
      const caseData = await CrimeVisionAPI.getCase(caseId);
      AppState.currentCaseId = caseId;
      AppState.currentCase = caseData;
      AppState.fusion = caseData.fusion || null;

      if (elements.caseSelect) {
        elements.caseSelect.value = caseId;
      }
      if (elements.headerCaseBadge) {
        elements.headerCaseBadge.textContent = `Case: ${caseData.case_id}`;
      }
      if (elements.caseTitleDisplay) {
        elements.caseTitleDisplay.textContent = caseData.title;
      }
      if (elements.caseDescDisplay) {
        elements.caseDescDisplay.textContent = caseData.description || 'No description provided.';
      }

      resetChatToCase(caseData.case_id);

      renderEvidenceLocker(caseData.evidence || []);
      const activeTimelineEvents = (AppState.fusion && AppState.fusion.events && AppState.fusion.events.length > 0)
        ? AppState.fusion.events
        : (caseData.timeline_events || []);
      renderCaseTimeline(activeTimelineEvents);
      renderFusionFindings(AppState.fusion);

      // Render case incident hypothesis
      resetHypothesisUI();
      AppState.caseHypothesis = caseData.hypothesis || null;
      if (caseData.hypothesis) {
        renderCaseHypothesis(caseData.hypothesis);
      } else {
        const hasAnalyzedEvidence = (caseData.evidence || []).some((e) => e.status === 'completed' && e.analysis);
        if (hasAnalyzedEvidence) {
          loadCaseHypothesis(caseId);
        } else {
          renderCaseHypothesis(null);
        }
      }

      updateOverviewMetrics();
      if (elements.btnExportPdf) {
        elements.btnExportPdf.disabled = false;
      }

      // Automatically inspect primary evidence if available
      const evList = caseData.evidence || [];
      const primaryEv = evList.find((e) => e.media_type === 'video' && e.status === 'completed' && e.analysis)
        || evList.find((e) => e.status === 'completed' && e.analysis)
        || evList[0];

      if (primaryEv && primaryEv.status === 'completed' && primaryEv.analysis) {
        loadEvidenceIntoViewer(primaryEv);
      } else {
        clearEvidenceViewer();
      }
    } catch (err) {
      showAlert(`Failed to load case ${caseId}: ${err.message}`, 'error');
    }
  }

  function resetChatToCase(caseId) {
    if (!elements.chatMessagesContainer) return;
    while (elements.chatMessagesContainer.firstChild) {
      elements.chatMessagesContainer.removeChild(elements.chatMessagesContainer.firstChild);
    }
    const initialMsg = document.createElement('div');
    initialMsg.className = 'chat-msg chat-msg-system';
    initialMsg.textContent = `Ready for ${caseId}. Ask questions about timeline events, person appearances, objects, or request investigative hypotheses.`;
    elements.chatMessagesContainer.appendChild(initialMsg);
  }

  function bindCaseEvents() {
    if (elements.caseSelect) {
      elements.caseSelect.addEventListener('change', (e) => {
        switchCase(e.target.value);
      });
    }

    if (elements.btnOpenNewCase) {
      elements.btnOpenNewCase.addEventListener('click', () => {
        elements.inputNewCaseTitle.value = '';
        elements.inputNewCaseId.value = '';
        elements.inputNewCaseDesc.value = '';
        clearModalAlert();
        elements.newCaseModal.style.display = 'flex';
      });
    }

    if (elements.btnCloseModal) {
      elements.btnCloseModal.addEventListener('click', () => {
        elements.newCaseModal.style.display = 'none';
      });
    }

    if (elements.btnCancelNewCase) {
      elements.btnCancelNewCase.addEventListener('click', () => {
        elements.newCaseModal.style.display = 'none';
      });
    }

    if (elements.btnSubmitNewCase) {
      elements.btnSubmitNewCase.addEventListener('click', handleCreateCaseSubmit);
    }
  }

  async function handleCreateCaseSubmit() {
    const title = elements.inputNewCaseTitle.value.trim();
    const caseId = elements.inputNewCaseId.value.trim() || null;
    const desc = elements.inputNewCaseDesc.value.trim();

    if (!title) {
      showModalAlert('Please enter a case title.', 'error');
      return;
    }

    try {
      const created = await CrimeVisionAPI.createCase(title, desc, caseId);
      elements.newCaseModal.style.display = 'none';
      showAlert(`Case ${created.case_id} created successfully.`, 'info');
      await loadCasesList();
      await switchCase(created.case_id);
    } catch (err) {
      showModalAlert(err.message || 'Failed to create case', 'error');
    }
  }

  function showModalAlert(message, type) {
    clearModalAlert();
    const div = document.createElement('div');
    div.className = `alert alert-${type}`;
    div.textContent = message;
    elements.modalAlertBox.appendChild(div);
  }

  function clearModalAlert() {
    while (elements.modalAlertBox.firstChild) {
      elements.modalAlertBox.removeChild(elements.modalAlertBox.firstChild);
    }
  }

  // -------------------------------------------------------------------------
  // Central Evidence Navigation Helper for Fusion & Findings
  // -------------------------------------------------------------------------
  function navigateToEvidenceSource(evidenceId, timestamp) {
    if (!evidenceId) return;
    const evItem = (AppState.currentCase?.evidence || []).find((e) => e.evidence_id === evidenceId);
    if (evItem) {
      loadEvidenceIntoViewer(evItem);
      if (typeof timestamp === 'number' && (evItem.media_type === 'video' || evItem.media_type === 'audio')) {
        setTimeout(() => {
          if (evItem.media_type === 'video' && elements.videoPlayerPreview) {
            elements.videoPlayerPreview.currentTime = timestamp;
            elements.videoPlayerPreview.play().catch(() => {});
            elements.videoPlayerPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          } else if (evItem.media_type === 'audio' && elements.audioPlayer) {
            elements.audioPlayer.currentTime = timestamp;
            elements.audioPlayer.play().catch(() => {});
          }
        }, 150);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Multimodal Fusion Event Binding & Rendering (Epic 6.1 & Investigator Redesign)
  // -------------------------------------------------------------------------
  function bindFusionEvents() {
    const runFusionAction = async () => {
      if (!AppState.currentCaseId) {
        showAlert('Select an active case first.', 'error');
        return;
      }
      if (elements.btnRunFusion) {
        elements.btnRunFusion.disabled = true;
        elements.btnRunFusion.textContent = '⚡ Fusing...';
      }
      if (elements.btnReFuse) {
        elements.btnReFuse.disabled = true;
        elements.btnReFuse.textContent = '⚡ Fusing...';
      }

      try {
        const fusionResult = await CrimeVisionAPI.fuseCase(AppState.currentCaseId);
        AppState.fusion = fusionResult;
        if (AppState.currentCase) {
          AppState.currentCase.fusion = fusionResult;
        }
        renderFusionFindings(fusionResult);
        const activeEvents = (fusionResult.events && fusionResult.events.length > 0)
          ? fusionResult.events
          : (AppState.currentCase ? AppState.currentCase.timeline_events || [] : []);
        renderCaseTimeline(activeEvents);
        updateOverviewMetrics();
        await evaluateCaseHypothesisAction(true);
        showAlert(
          `Cross-modal fusion completed: ${(fusionResult.events || []).length} events, ${(fusionResult.relationships || []).length} relationships.`,
          'info'
        );
      } catch (err) {
        showAlert(`Cross-modal fusion failed: ${err.message}`, 'error');
      } finally {
        if (elements.btnRunFusion) {
          elements.btnRunFusion.disabled = false;
          elements.btnRunFusion.textContent = '⚡ Fuse Evidence';
        }
        if (elements.btnReFuse) {
          elements.btnReFuse.disabled = false;
          elements.btnReFuse.textContent = '⚡ Re-Fuse';
        }
      }
    };

    if (elements.btnRunFusion) {
      elements.btnRunFusion.addEventListener('click', runFusionAction);
    }
    if (elements.btnReFuse) {
      elements.btnReFuse.addEventListener('click', runFusionAction);
    }
  }

  function renderFusionFindings(fusionData) {
    if (!elements.fusionSummaryCard) return;

    if (!fusionData) {
      if (elements.fusionMainContainer) elements.fusionMainContainer.style.display = 'none';
      elements.fusionSummaryCard.style.display = 'none';
      if (elements.fusionRelationshipsCard) elements.fusionRelationshipsCard.style.display = 'none';
      if (elements.fusionEmptyNotice) elements.fusionEmptyNotice.style.display = 'block';
      return;
    }

    if (elements.fusionEmptyNotice) elements.fusionEmptyNotice.style.display = 'none';
    if (elements.fusionMainContainer) elements.fusionMainContainer.style.display = 'flex';
    elements.fusionSummaryCard.style.display = 'flex';

    const events = fusionData.events || [];
    const relationships = fusionData.relationships || [];
    const anchors = fusionData.anchors || [];
    const unalignedEvents = fusionData.unaligned_events || [];

    // -------------------------------------------------------------------------
    // SECTION A: Overview, Accurately Calculated Metrics & Alignment Status
    // -------------------------------------------------------------------------
    const isTemporallyAligned = unalignedEvents.length === 0 && events.length > 0;
    if (elements.fusionAlignmentBadge) {
      if (isTemporallyAligned) {
        elements.fusionAlignmentBadge.textContent = '⏱ Aligned Common Timeline';
        elements.fusionAlignmentBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        elements.fusionAlignmentBadge.style.color = '#34d399';
        elements.fusionAlignmentBadge.title = 'All observations share synchronized timeline coordinates.';
      } else {
        elements.fusionAlignmentBadge.textContent = '⚠️ Independent Time Bases';
        elements.fusionAlignmentBadge.style.background = 'rgba(245, 158, 11, 0.15)';
        elements.fusionAlignmentBadge.style.color = '#fbbf24';
        elements.fusionAlignmentBadge.title = 'Sources operate on independent relative timers without common absolute UTC synchronization.';
      }
    }

    if (elements.fusionAnchorsBadge) {
      elements.fusionAnchorsBadge.textContent = `${anchors.length} Key Anchor${anchors.length !== 1 ? 's' : ''}`;
    }

    // Accurately calculated metrics
    const conflicts = relationships.filter((r) => r.relationship_type === 'conflict');
    const itemsForReview = events.filter((e) =>
      (e.certainty_level === 'low') ||
      (typeof e.source_confidence === 'number' && e.source_confidence < 0.60) ||
      (e.alignment_status !== 'aligned_common_timeline')
    );

    if (elements.fusionMetricRelations) {
      elements.fusionMetricRelations.textContent = String(relationships.length);
    }
    if (elements.fusionMetricConflicts) {
      elements.fusionMetricConflicts.textContent = String(conflicts.length);
      const conflictParent = elements.fusionMetricConflicts.closest('.fusion-metric-item');
      if (conflictParent) {
        if (conflicts.length > 0) {
          conflictParent.classList.add('has-conflicts');
        } else {
          conflictParent.classList.remove('has-conflicts');
        }
      }
    }
    if (elements.fusionMetricReview) {
      elements.fusionMetricReview.textContent = String(itemsForReview.length);
    }
    if (elements.fusionMetricEvents) {
      elements.fusionMetricEvents.textContent = String(events.length);
    }

    // Synthesis Text
    if (elements.fusionSynthesisText) {
      const evidenceList = AppState.currentCase?.evidence || [];
      const vCount = evidenceList.filter((e) => e.media_type === 'video').length;
      const aCount = evidenceList.filter((e) => e.media_type === 'audio').length;
      const iCount = evidenceList.filter((e) => e.media_type === 'image').length;
      elements.fusionSynthesisText.textContent =
        `Synthesized ${events.length} multi-source forensic observation${events.length !== 1 ? 's' : ''} across ${evidenceList.length} evidence file${evidenceList.length !== 1 ? 's' : ''} ` +
        `(${vCount} video, ${aCount} audio, ${iCount} image). Established ${relationships.length} distinct cross-modal relationship${relationships.length !== 1 ? 's' : ''} ` +
        `and flagged ${conflicts.length} evidentiary discrepancy/conflict${conflicts.length !== 1 ? 's' : ''}.`;
    }

    // Evidence Gaps & Alignment Banner
    if (elements.fusionGapsBanner && elements.fusionGapsContent) {
      const gaps = [];
      if (!isTemporallyAligned && unalignedEvents.length > 0) {
        gaps.push(`<strong>Independent Time Bases:</strong> Evidence files lack synchronized UTC clocks. Co-occurrences indicate intra-media sequence, not confirmed real-time concurrence.`);
      }
      const evidenceList = AppState.currentCase?.evidence || [];
      const hasAudio = evidenceList.some((e) => e.media_type === 'audio');
      const hasVideo = evidenceList.some((e) => e.media_type === 'video');
      if (!hasAudio) {
        gaps.push(`<strong>Missing Audio Modality:</strong> No witness testimony or acoustic records available for spoken corroboration.`);
      }
      if (!hasVideo) {
        gaps.push(`<strong>Missing Video Modality:</strong> No continuous visual stream available for continuous tracking.`);
      }
      if (itemsForReview.length > 0) {
        gaps.push(`<strong>Forensic Review Required:</strong> ${itemsForReview.length} observation(s) have low model confidence (< 0.60) or unverified time bases.`);
      }

      if (gaps.length > 0) {
        elements.fusionGapsContent.innerHTML = gaps.join('<br style="margin-bottom:0.25rem;">');
        elements.fusionGapsBanner.style.display = 'flex';
      } else {
        elements.fusionGapsBanner.style.display = 'none';
      }
    }

    // -------------------------------------------------------------------------
    // SECTION B: Key Findings (Grouped by investigative meaning)
    // -------------------------------------------------------------------------
    if (elements.fusionKeyFindingsList) {
      while (elements.fusionKeyFindingsList.firstChild) {
        elements.fusionKeyFindingsList.removeChild(elements.fusionKeyFindingsList.firstChild);
      }

      // Groupings
      const witnessFindings = events.filter((e) =>
        e.modality === 'audio' ||
        (e.event_type && (e.event_type.includes('speech') || e.event_type.includes('statement') || e.event_type.includes('claim')))
      );

      const visualFindings = events.filter((e) =>
        (e.modality === 'video' || e.modality === 'image') &&
        (e.is_anchor || (e.observed_labels && e.observed_labels.some((l) => ['person', 'knife', 'gun', 'weapon'].includes(l))))
      );

      // Representative visual findings (limit to 6 representative distinct anchors/frames)
      const repVisualFindings = visualFindings.slice(0, 6);

      const groups = [];

      if (conflicts.length > 0) {
        groups.push({
          title: '⚠️ Discrepancies & Contradictions',
          icon: '⚠️',
          classMod: 'finding-conflict',
          items: conflicts.map((c) => ({
            desc: c.explanation || c.description,
            chipA: {
              modality: c.source_a_modality || 'media',
              filename: c.source_a_filename || 'Source A',
              timestamp: c.source_a_timestamp,
              evId: c.source_a_evidence_id,
            },
            chipB: {
              modality: c.source_b_modality || 'media',
              filename: c.source_b_filename || 'Source B',
              timestamp: c.source_b_timestamp,
              evId: c.source_b_evidence_id,
            },
            badge: 'Conflict Flagged',
            badgeClass: 'badge-rel-conflict',
          })),
        });
      }

      if (witnessFindings.length > 0) {
        groups.push({
          title: '🗣 Witness Claims & Audio Statements',
          icon: '🗣',
          classMod: 'finding-witness',
          items: witnessFindings.slice(0, 5).map((w) => ({
            desc: w.description || `Audio statement recorded in ${w.original_filename}`,
            chipA: {
              modality: 'audio',
              filename: w.original_filename,
              timestamp: w.timestamp_normalized,
              evId: w.evidence_id,
            },
            badge: w.extracted_entities && w.extracted_entities.length > 0 ? w.extracted_entities.join(', ') : 'Statement',
            badgeClass: 'badge-case',
          })),
        });
      }

      if (repVisualFindings.length > 0) {
        groups.push({
          title: '👁 Key Visual Observations & Detections',
          icon: '👁',
          classMod: 'finding-visual',
          items: repVisualFindings.map((v) => ({
            desc: v.description || `Visual observation in ${v.original_filename}`,
            chipA: {
              modality: v.modality,
              filename: v.original_filename,
              timestamp: v.timestamp_normalized,
              evId: v.evidence_id,
            },
            badge: v.observed_labels && v.observed_labels.length > 0 ? v.observed_labels.slice(0, 3).join(', ') : v.event_type,
            badgeClass: 'badge-case',
          })),
        });
      }

      let totalFindingItems = 0;
      groups.forEach((grp) => {
        const grpWrap = document.createElement('div');
        grpWrap.className = 'fusion-finding-group';

        const titleEl = document.createElement('div');
        titleEl.className = 'fusion-finding-group-title';
        titleEl.textContent = grp.title;
        grpWrap.appendChild(titleEl);

        grp.items.forEach((item) => {
          totalFindingItems++;
          const itemEl = document.createElement('div');
          itemEl.className = `fusion-finding-item ${grp.classMod}`;

          const topRow = document.createElement('div');
          topRow.className = 'fusion-finding-top';

          const chipsWrap = document.createElement('div');
          chipsWrap.style.display = 'flex';
          chipsWrap.style.alignItems = 'center';
          chipsWrap.style.gap = '0.35rem';
          chipsWrap.style.flexWrap = 'wrap';

          if (item.chipA) {
            const chipA = document.createElement('button');
            chipA.type = 'button';
            chipA.className = 'fusion-source-chip';
            const mIcon = item.chipA.modality === 'audio' ? '🔊' : (item.chipA.modality === 'image' ? '🖼️' : '🎥');
            const tsStr = typeof item.chipA.timestamp === 'number' ? ` @ ${item.chipA.timestamp.toFixed(2)}s` : '';
            chipA.textContent = `${mIcon} ${item.chipA.filename}${tsStr}`;
            chipA.title = 'Click to inspect in primary canvas';
            chipA.addEventListener('click', (e) => {
              e.stopPropagation();
              navigateToEvidenceSource(item.chipA.evId, item.chipA.timestamp);
            });
            chipsWrap.appendChild(chipA);
          }

          if (item.chipB) {
            const vsSpan = document.createElement('span');
            vsSpan.className = 'fusion-vs-divider';
            vsSpan.textContent = 'vs';
            chipsWrap.appendChild(vsSpan);

            const chipB = document.createElement('button');
            chipB.type = 'button';
            chipB.className = 'fusion-source-chip';
            const mIconB = item.chipB.modality === 'audio' ? '🔊' : (item.chipB.modality === 'image' ? '🖼️' : '🎥');
            const tsStrB = typeof item.chipB.timestamp === 'number' ? ` @ ${item.chipB.timestamp.toFixed(2)}s` : '';
            chipB.textContent = `${mIconB} ${item.chipB.filename}${tsStrB}`;
            chipB.title = 'Click to inspect in primary canvas';
            chipB.addEventListener('click', (e) => {
              e.stopPropagation();
              navigateToEvidenceSource(item.chipB.evId, item.chipB.timestamp);
            });
            chipsWrap.appendChild(chipB);
          }

          topRow.appendChild(chipsWrap);

          if (item.badge) {
            const b = document.createElement('span');
            b.className = `badge ${item.badgeClass || 'badge-case'}`;
            b.style.fontSize = '0.68rem';
            b.textContent = item.badge;
            topRow.appendChild(b);
          }

          itemEl.appendChild(topRow);

          const descEl = document.createElement('p');
          descEl.className = 'fusion-finding-desc';
          descEl.textContent = item.desc;
          itemEl.appendChild(descEl);

          grpWrap.appendChild(itemEl);
        });

        elements.fusionKeyFindingsList.appendChild(grpWrap);
      });

      if (elements.fusionFindingsCountBadge) {
        elements.fusionFindingsCountBadge.textContent = `${totalFindingItems} Finding${totalFindingItems !== 1 ? 's' : ''}`;
      }
    }

    // -------------------------------------------------------------------------
    // SECTION C: Evidence Relationships (Readable Cards with Source Comparison)
    // -------------------------------------------------------------------------
    if (elements.fusionRelationshipsCard) {
      if (relationships.length === 0) {
        elements.fusionRelationshipsCard.style.display = 'none';
      } else {
        elements.fusionRelationshipsCard.style.display = 'flex';
        if (elements.fusionRelCountBadge) {
          elements.fusionRelCountBadge.textContent = `${relationships.length} Link${relationships.length !== 1 ? 's' : ''}`;
        }
        if (elements.fusionRelationshipsList) {
          while (elements.fusionRelationshipsList.firstChild) {
            elements.fusionRelationshipsList.removeChild(elements.fusionRelationshipsList.firstChild);
          }

          relationships.forEach((rel) => {
            const relCard = document.createElement('div');
            const rType = rel.relationship_type || 'relationship';
            relCard.className = `fusion-rel-card rel-type-${rType}`;

            // Header Row: Badges & ID
            const headerRow = document.createElement('div');
            headerRow.className = 'fusion-rel-header';

            const badgeGroup = document.createElement('div');
            badgeGroup.className = 'fusion-rel-badges';

            const typeBadge = document.createElement('span');
            typeBadge.className = 'badge';
            typeBadge.style.fontSize = '0.72rem';
            typeBadge.style.fontWeight = '700';

            if (rType === 'conflict') {
              typeBadge.textContent = '⚠️ Conflict or Discrepancy';
              typeBadge.classList.add('badge-rel-conflict');
            } else if (rType === 'thematic_match') {
              typeBadge.textContent = '🔄 Thematic Match Only';
              typeBadge.classList.add('badge-rel-thematic');
            } else if (rType === 'qualified_corroboration') {
              typeBadge.textContent = '✓ Qualified Corroboration';
              typeBadge.classList.add('badge-rel-corroboration');
            } else if (rType === 'corroboration') {
              typeBadge.textContent = '✓ Corroboration';
              typeBadge.classList.add('badge-rel-corroboration');
            } else if (rType === 'potentially_related' || rType === 'potential_consistency') {
              typeBadge.textContent = 'ℹ️ Potentially Related';
              typeBadge.classList.add('badge-rel-potential');
            } else {
              typeBadge.textContent = `• ${(rType).replace(/_/g, ' ')}`;
              typeBadge.classList.add('badge-rel-insufficient');
            }
            badgeGroup.appendChild(typeBadge);

            if (rel.time_alignment_basis) {
              const alignChip = document.createElement('span');
              alignChip.className = 'badge badge-case';
              alignChip.style.fontSize = '0.66rem';
              alignChip.textContent = rel.time_alignment_basis === 'synchronized_timeline' ? '⏱ Synchronized' : '⚠️ Semantic (Unaligned)';
              badgeGroup.appendChild(alignChip);
            }

            const idSpan = document.createElement('span');
            idSpan.style.fontSize = '0.7rem';
            idSpan.style.color = 'var(--text-dim)';
            idSpan.textContent = rel.relationship_id || rel.rel_id || '';

            headerRow.appendChild(badgeGroup);
            headerRow.appendChild(idSpan);
            relCard.appendChild(headerRow);

            // Source Comparison Row
            const compWrap = document.createElement('div');
            compWrap.className = 'fusion-comparison-wrap';

            // Source A Chip
            const sA = document.createElement('button');
            sA.type = 'button';
            sA.className = 'fusion-comparison-chip';
            const modA = (rel.source_a_modality || 'media').toLowerCase();
            const iconA = modA === 'audio' ? '🔊' : (modA === 'image' ? '🖼️' : '🎥');
            const tsA = typeof rel.source_a_timestamp === 'number' ? ` @ ${rel.source_a_timestamp.toFixed(2)}s` : '';
            sA.textContent = `${iconA} ${rel.source_a_filename || 'Source A'}${tsA}`;
            sA.title = 'Click to inspect original evidence';
            sA.addEventListener('click', (e) => {
              e.stopPropagation();
              navigateToEvidenceSource(rel.source_a_evidence_id, rel.source_a_timestamp);
            });
            compWrap.appendChild(sA);

            const vsEl = document.createElement('span');
            vsEl.className = 'fusion-vs-divider';
            vsEl.textContent = 'correlated with';
            compWrap.appendChild(vsEl);

            // Source B Chip
            const sB = document.createElement('button');
            sB.type = 'button';
            sB.className = 'fusion-comparison-chip';
            const modB = (rel.source_b_modality || 'media').toLowerCase();
            const iconB = modB === 'audio' ? '🔊' : (modB === 'image' ? '🖼️' : '🎥');
            const tsB = typeof rel.source_b_timestamp === 'number' ? ` @ ${rel.source_b_timestamp.toFixed(2)}s` : '';
            sB.textContent = `${iconB} ${rel.source_b_filename || 'Source B'}${tsB}`;
            sB.title = 'Click to inspect original evidence';
            sB.addEventListener('click', (e) => {
              e.stopPropagation();
              navigateToEvidenceSource(rel.source_b_evidence_id, rel.source_b_timestamp);
            });
            compWrap.appendChild(sB);

            relCard.appendChild(compWrap);

            // Plain-Language Reason / Explanation
            const reason = rel.explanation || rel.description;
            if (reason) {
              const reasonP = document.createElement('p');
              reasonP.className = 'fusion-rel-reason';
              reasonP.textContent = reason;
              relCard.appendChild(reasonP);
            }

            // Limitations / Uncertainties
            const caveats = rel.uncertainties || rel.caveats;
            if (caveats && caveats.length > 0) {
              const caveatBox = document.createElement('div');
              caveatBox.className = 'fusion-rel-caveat';
              const caveatList = Array.isArray(caveats) ? caveats.join(' • ') : caveats;
              caveatBox.textContent = `Limitations: ${caveatList}`;
              relCard.appendChild(caveatBox);
            }

            elements.fusionRelationshipsList.appendChild(relCard);
          });
        }
      }
    }

    // -------------------------------------------------------------------------
    // SECTION D: Technical Evidence & Fusion Audit Trail (Collapsed by default)
    // -------------------------------------------------------------------------
    if (elements.fusionNarrativeText) {
      elements.fusionNarrativeText.textContent =
        fusionData.grounded_narrative || 'Multimodal synthesis completed across recorded evidence.';
    }

    if (elements.fusionHashMeta) {
      const hIndex = fusionData.hash_index || {};
      const tBuckets = hIndex.time_buckets_count ?? (hIndex.time_buckets ? Object.keys(hIndex.time_buckets).length : 0);
      const termBuckets = hIndex.term_buckets_count ?? (hIndex.term_buckets ? Object.keys(hIndex.term_buckets).length : 0);
      const schemaVer = fusionData.schema_version || '1.0.0';
      const fusedAt = fusionData.fused_at || 'Recorded';
      elements.fusionHashMeta.innerHTML =
        `Deterministic SHA-256 Index: ${tBuckets} temporal buckets, ${termBuckets} term buckets | Schema: v${schemaVer} | Fused: ${fusedAt}`;
    }

    if (elements.fusionEventsTableWrap) {
      while (elements.fusionEventsTableWrap.firstChild) {
        elements.fusionEventsTableWrap.removeChild(elements.fusionEventsTableWrap.firstChild);
      }

      if (events.length > 0) {
        const table = document.createElement('table');
        table.className = 'fusion-events-table';

        const thead = document.createElement('thead');
        thead.innerHTML = `
          <tr>
            <th>Event ID</th>
            <th>Modality</th>
            <th>Source File</th>
            <th>Timestamp</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Certainty</th>
          </tr>
        `;
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        events.forEach((ev) => {
          const row = document.createElement('tr');
          row.style.cursor = 'pointer';
          row.title = 'Click to inspect in primary canvas';
          const ts = typeof ev.timestamp_normalized === 'number' ? `${ev.timestamp_normalized.toFixed(2)}s` : '--';
          const conf = typeof ev.source_confidence === 'number' ? `${(ev.source_confidence * 100).toFixed(0)}%` : '--';
          const cert = (ev.certainty_level || 'medium').toUpperCase();

          row.innerHTML = `
            <td><code>${ev.event_id || '--'}</code></td>
            <td>${ev.modality || '--'}</td>
            <td>${ev.original_filename || '--'}</td>
            <td>${ts}</td>
            <td>${(ev.event_type || '--').replace(/_/g, ' ')}</td>
            <td>${conf}</td>
            <td><span class="badge badge-case">${cert}</span></td>
          `;

          row.addEventListener('click', () => {
            navigateToEvidenceSource(ev.evidence_id, ev.timestamp_normalized);
          });

          tbody.appendChild(row);
        });
        table.appendChild(tbody);
        elements.fusionEventsTableWrap.appendChild(table);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Incident Hypothesis Management & Rendering
  // -------------------------------------------------------------------------
  function bindHypothesisEvents() {
    if (elements.btnRefreshHypothesis) {
      elements.btnRefreshHypothesis.addEventListener('click', () => {
        evaluateCaseHypothesisAction(true);
      });
    }
    if (elements.btnRetryHypothesis) {
      elements.btnRetryHypothesis.addEventListener('click', () => {
        evaluateCaseHypothesisAction(true);
      });
    }
  }

  function resetHypothesisUI() {
    if (elements.hypothesisLoading) elements.hypothesisLoading.style.display = 'none';
    if (elements.hypothesisError) elements.hypothesisError.style.display = 'none';
    if (elements.hypothesisContent) elements.hypothesisContent.style.display = 'none';
    if (elements.hypothesisEmpty) elements.hypothesisEmpty.style.display = 'block';
    if (elements.hypothesisEmptyTitle) elements.hypothesisEmptyTitle.textContent = 'No Incident Hypothesis Evaluated Yet';
    if (elements.hypothesisEmptySub) elements.hypothesisEmptySub.textContent = 'Ingest case evidence (video, audio, or images) and click Evaluate Hypothesis to synthesize candidate incident types grounded in recorded findings.';
    if (elements.hypothesisStatusBadge) {
      elements.hypothesisStatusBadge.textContent = 'Awaiting Evaluation';
      elements.hypothesisStatusBadge.className = 'badge badge-case';
    }
  }

  async function loadCaseHypothesis(caseId) {
    if (!caseId) return;
    try {
      const hyp = await CrimeVisionAPI.getCaseHypothesis(caseId);
      AppState.caseHypothesis = hyp;
      renderCaseHypothesis(hyp);
    } catch (err) {
      renderHypothesisError(err.message);
    }
  }

  async function evaluateCaseHypothesisAction(forceRefresh = true) {
    if (!AppState.currentCaseId) {
      showAlert('Select an active case first.', 'error');
      return;
    }

    if (elements.hypothesisLoading) elements.hypothesisLoading.style.display = 'block';
    if (elements.hypothesisEmpty) elements.hypothesisEmpty.style.display = 'none';
    if (elements.hypothesisError) elements.hypothesisError.style.display = 'none';
    if (elements.hypothesisContent) elements.hypothesisContent.style.display = 'none';
    if (elements.btnRefreshHypothesis) {
      elements.btnRefreshHypothesis.disabled = true;
      elements.btnRefreshHypothesis.textContent = '⚡ Evaluating...';
    }
    if (elements.hypothesisStatusBadge) {
      elements.hypothesisStatusBadge.textContent = 'Evaluating...';
      elements.hypothesisStatusBadge.className = 'badge badge-case';
    }

    try {
      const hyp = await CrimeVisionAPI.evaluateCaseHypothesis(AppState.currentCaseId);
      AppState.caseHypothesis = hyp;
      if (AppState.currentCase) {
        AppState.currentCase.hypothesis = hyp;
      }
      renderCaseHypothesis(hyp);
    } catch (err) {
      renderHypothesisError(err.message);
    } finally {
      if (elements.btnRefreshHypothesis) {
        elements.btnRefreshHypothesis.disabled = false;
        elements.btnRefreshHypothesis.textContent = '⚡ Evaluate Hypothesis';
      }
    }
  }

  function renderHypothesisError(message) {
    if (elements.hypothesisLoading) elements.hypothesisLoading.style.display = 'none';
    if (elements.hypothesisEmpty) elements.hypothesisEmpty.style.display = 'none';
    if (elements.hypothesisContent) elements.hypothesisContent.style.display = 'none';
    if (elements.hypothesisError) elements.hypothesisError.style.display = 'block';
    if (elements.hypothesisErrorSub) elements.hypothesisErrorSub.textContent = message || 'Evaluation failed.';
    if (elements.hypothesisStatusBadge) {
      elements.hypothesisStatusBadge.textContent = 'Evaluation Error';
      elements.hypothesisStatusBadge.className = 'badge badge-status-error';
    }
  }

  function renderCaseHypothesis(hyp) {
    if (!elements.hypothesisContent) return;

    if (!hyp || hyp.status === 'insufficient_evidence' || (hyp.primary_hypothesis && hyp.primary_hypothesis.plausibility === 'Undetermined')) {
      if (elements.hypothesisLoading) elements.hypothesisLoading.style.display = 'none';
      if (elements.hypothesisError) elements.hypothesisError.style.display = 'none';
      if (elements.hypothesisContent) elements.hypothesisContent.style.display = 'none';
      if (elements.hypothesisEmpty) elements.hypothesisEmpty.style.display = 'block';
      if (elements.hypothesisEmptyTitle) {
        elements.hypothesisEmptyTitle.textContent = (hyp && hyp.incident_type) || 'Undetermined / Insufficient Evidence';
      }
      if (elements.hypothesisEmptySub) {
        elements.hypothesisEmptySub.textContent = (hyp && hyp.narrative) || 'No analyzed evidence is currently recorded in this case locker to formulate hypotheses.';
      }
      if (elements.hypothesisStatusBadge) {
        elements.hypothesisStatusBadge.textContent = 'Insufficient Evidence';
        elements.hypothesisStatusBadge.className = 'badge badge-status-neutral';
      }
      return;
    }

    if (elements.hypothesisLoading) elements.hypothesisLoading.style.display = 'none';
    if (elements.hypothesisEmpty) elements.hypothesisEmpty.style.display = 'none';
    if (elements.hypothesisError) elements.hypothesisError.style.display = 'none';
    elements.hypothesisContent.style.display = 'flex';

    if (elements.hypothesisStatusBadge) {
      elements.hypothesisStatusBadge.textContent = 'Evaluated';
      elements.hypothesisStatusBadge.className = 'badge badge-status-ready';
    }

    // 1. Primary Candidate Incident
    const primaryType = hyp.incident_type || (hyp.primary_hypothesis && hyp.primary_hypothesis.incident_type) || 'Candidate Incident';
    if (elements.hypPrimaryType) elements.hypPrimaryType.textContent = primaryType;

    const plausibility = hyp.plausibility || (hyp.primary_hypothesis && hyp.primary_hypothesis.plausibility) || 'Limited';
    if (elements.hypPlausibilityBadge) {
      elements.hypPlausibilityBadge.textContent = plausibility;
      const slug = plausibility.toLowerCase().replace(/[^a-z0-9]/g, '-');
      elements.hypPlausibilityBadge.className = `badge hyp-plausibility-badge hyp-plausibility-${slug}`;
    }

    const narrative = hyp.narrative || (hyp.primary_hypothesis && hyp.primary_hypothesis.narrative) || '';
    if (elements.hypPrimaryNarrative) elements.hypPrimaryNarrative.textContent = narrative;

    // 2. Alternative Candidate Hypotheses
    const alts = hyp.alternative_hypotheses || [];
    if (elements.hypAlternativesGroup) {
      if (alts.length > 0) {
        elements.hypAlternativesGroup.style.display = 'block';
        if (elements.hypAlternativesList) {
          while (elements.hypAlternativesList.firstChild) {
            elements.hypAlternativesList.removeChild(elements.hypAlternativesList.firstChild);
          }
          alts.forEach((alt) => {
            const card = document.createElement('div');
            card.className = 'hyp-alt-card';

            const header = document.createElement('div');
            header.className = 'hyp-alt-header';

            const title = document.createElement('h5');
            title.className = 'hyp-alt-title';
            title.textContent = alt.title || alt.incident_type || 'Alternative Hypothesis';

            const badge = document.createElement('span');
            const altPlaus = alt.plausibility || 'Moderate';
            badge.textContent = altPlaus;
            const altSlug = altPlaus.toLowerCase().replace(/[^a-z0-9]/g, '-');
            badge.className = `badge hyp-plausibility-badge hyp-plausibility-${altSlug}`;

            header.appendChild(title);
            header.appendChild(badge);

            const desc = document.createElement('p');
            desc.className = 'hyp-alt-narrative';
            desc.textContent = alt.narrative || '';

            card.appendChild(header);
            card.appendChild(desc);
            elements.hypAlternativesList.appendChild(card);
          });
        }
      } else {
        elements.hypAlternativesGroup.style.display = 'none';
      }
    }

    // 3. Supporting Evidence
    const supp = hyp.supporting_evidence || (hyp.primary_hypothesis && hyp.primary_hypothesis.supporting_evidence) || [];
    if (elements.hypSupportingList) {
      while (elements.hypSupportingList.firstChild) {
        elements.hypSupportingList.removeChild(elements.hypSupportingList.firstChild);
      }
      if (supp.length > 0) {
        supp.forEach((s) => {
          const li = document.createElement('li');
          li.textContent = typeof s === 'string' ? s : JSON.stringify(s);
          elements.hypSupportingList.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.textContent = 'Human presence verified across recorded frames; no specific crime actions directly observed.';
        elements.hypSupportingList.appendChild(li);
      }
    }

    // 4. Contradictory or Conflicting Evidence
    const contra = hyp.contradictions || (hyp.primary_hypothesis && hyp.primary_hypothesis.contradicting_evidence) || [];
    const validContra = contra.filter((c) => c && c !== 'None identified in available evidence');
    if (elements.hypContradictionsGroup) {
      if (validContra.length > 0) {
        elements.hypContradictionsGroup.style.display = 'block';
        if (elements.hypContradictionsList) {
          while (elements.hypContradictionsList.firstChild) {
            elements.hypContradictionsList.removeChild(elements.hypContradictionsList.firstChild);
          }
          validContra.forEach((c) => {
            const li = document.createElement('li');
            li.textContent = c;
            elements.hypContradictionsList.appendChild(li);
          });
        }
      } else {
        elements.hypContradictionsGroup.style.display = 'none';
      }
    }

    // 5. Evidence Gaps & Forensic Limitations
    const gaps = hyp.evidence_gaps || (hyp.primary_hypothesis && hyp.primary_hypothesis.uncertainty_factors) || [];
    if (elements.hypGapsList) {
      while (elements.hypGapsList.firstChild) {
        elements.hypGapsList.removeChild(elements.hypGapsList.firstChild);
      }
      if (gaps.length > 0) {
        gaps.forEach((g) => {
          const li = document.createElement('li');
          li.textContent = g;
          elements.hypGapsList.appendChild(li);
        });
      } else {
        const li = document.createElement('li');
        li.textContent = 'Single camera perspective; lack of persistent multi-camera tracking.';
        elements.hypGapsList.appendChild(li);
      }
    }

    // 6. Overall Assessment
    const assessment = hyp.overall_assessment || (hyp.primary_hypothesis && hyp.primary_hypothesis.narrative) || 'Evidence evaluation completed.';
    if (elements.hypOverallAssessment) {
      elements.hypOverallAssessment.textContent = assessment;
    }
  }

  // -------------------------------------------------------------------------
  // Case Evidence Locker Rendering
  // -------------------------------------------------------------------------
  function renderEvidenceLocker(evidenceList) {
    const count = evidenceList.length;
    elements.caseEvidenceCountBadge.textContent = `${count} Item${count !== 1 ? 's' : ''}`;

    while (elements.caseEvidenceList.firstChild) {
      elements.caseEvidenceList.removeChild(elements.caseEvidenceList.firstChild);
    }

    if (count === 0) {
      elements.evidenceEmptyNotice.style.display = 'block';
      elements.caseEvidenceList.style.display = 'none';
      return;
    }

    elements.evidenceEmptyNotice.style.display = 'none';
    elements.caseEvidenceList.style.display = 'flex';

    evidenceList.forEach((ev) => {
      const item = document.createElement('div');
      item.className = 'evidence-locker-item';
      item.dataset.evidenceId = ev.evidence_id;
      if (AppState.selectedEvidenceId === ev.evidence_id) {
        item.classList.add('is-selected');
      }

      const left = document.createElement('div');
      left.className = 'evidence-item-left';

      const icon = document.createElement('span');
      icon.className = 'evidence-item-icon';
      icon.textContent = ev.media_type === 'audio' ? '🔊' : (ev.media_type === 'video' ? '🎬' : '🖼️');

      const info = document.createElement('div');
      info.className = 'evidence-item-info';

      const title = document.createElement('span');
      title.className = 'evidence-item-title';
      title.textContent = ev.original_filename;

      const sub = document.createElement('span');
      sub.className = 'evidence-item-sub';
      const sizeKb = ev.file_size ? `${(ev.file_size / 1024).toFixed(1)} KB` : '';
      sub.textContent = `${ev.evidence_id} • ${ev.media_type.toUpperCase()} ${sizeKb ? '• ' + sizeKb : ''}`;

      info.appendChild(title);
      info.appendChild(sub);
      left.appendChild(icon);
      left.appendChild(info);

      const actions = document.createElement('div');
      actions.className = 'evidence-item-actions';

      const statusBadge = document.createElement('span');
      statusBadge.className = `badge badge-status-${ev.status}`;
      statusBadge.textContent = ev.status === 'completed' ? 'Analyzed' : ev.status.toUpperCase();
      actions.appendChild(statusBadge);

      if (ev.status === 'completed' && ev.analysis) {
        const btnView = document.createElement('button');
        btnView.className = 'btn btn-secondary btn-xs btn-evidence-view';
        btnView.dataset.evidenceId = ev.evidence_id;
        btnView.textContent = 'View';
        btnView.addEventListener('click', () => loadEvidenceIntoViewer(ev));
        actions.appendChild(btnView);
      } else {
        const btnAnalyze = document.createElement('button');
        btnAnalyze.className = 'btn btn-primary btn-xs btn-evidence-analyze';
        btnAnalyze.dataset.evidenceId = ev.evidence_id;
        btnAnalyze.textContent = 'Analyze';
        btnAnalyze.addEventListener('click', async () => {
          btnAnalyze.disabled = true;
          btnAnalyze.textContent = '...';
          try {
            await CrimeVisionAPI.analyzeCaseEvidence(AppState.currentCaseId, ev.evidence_id);
            await switchCase(AppState.currentCaseId);
            showAlert(`Analysis completed for ${ev.original_filename}.`, 'info');
          } catch (err) {
            showAlert(`Analysis failed: ${err.message}`, 'error');
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = 'Analyze';
          }
        });
        actions.appendChild(btnAnalyze);
      }

      item.appendChild(left);
      item.appendChild(actions);
      elements.caseEvidenceList.appendChild(item);
    });
  }

  // -------------------------------------------------------------------------
  // Central Primary Media Inspection Stage (Decoupled from Ingestion Tabs)
  // -------------------------------------------------------------------------
  function loadEvidenceIntoViewer(evidenceItem) {
    if (!evidenceItem) return;
    AppState.selectedEvidenceId = evidenceItem.evidence_id;

    // Highlight selected card in locker
    if (elements.caseEvidenceList) {
      const allCards = elements.caseEvidenceList.querySelectorAll('.evidence-locker-item');
      allCards.forEach((c) => {
        if (c.dataset.evidenceId === evidenceItem.evidence_id) {
          c.classList.add('is-selected');
        } else {
          c.classList.remove('is-selected');
        }
      });
    }

    if (elements.inspectingFilename) {
      elements.inspectingFilename.textContent = evidenceItem.original_filename;
    }
    if (elements.inspectingBadge) {
      elements.inspectingBadge.textContent = evidenceItem.status === 'completed' ? 'Analyzed' : evidenceItem.status.toUpperCase();
      elements.inspectingBadge.className = `badge badge-status-${evidenceItem.status}`;
    }
    if (elements.mediaEmptyPlaceholder) {
      elements.mediaEmptyPlaceholder.style.display = 'none';
    }

    if (evidenceItem.media_type === 'audio') {
      if (elements.videoPlayerPreview) {
        elements.videoPlayerPreview.pause();
        elements.videoPlayerPreview.style.display = 'none';
      }
      if (elements.imageViewerContainer) elements.imageViewerContainer.style.display = 'none';
      if (elements.audioViewerContainer) elements.audioViewerContainer.style.display = 'flex';
      if (elements.inspectingMediaTypeIcon) elements.inspectingMediaTypeIcon.textContent = '🔊';

      if (elements.videoResultsPanel) elements.videoResultsPanel.style.display = 'none';
      if (elements.imageResultsPanel) elements.imageResultsPanel.style.display = 'none';
      if (elements.audioFactsPanel) elements.audioFactsPanel.style.display = 'block';

      const res = evidenceItem.analysis;
      if (res) {
        AppState.status = 'completed';
        AppState.evidence = {
          filename: evidenceItem.original_filename,
          duration: res.recording_duration_seconds,
        };
        AppState.transcription = res.transcription;
        AppState.timeline = res.timeline;
        AppState.events = (res.timeline && res.timeline.events) || [];
        renderState();
      }
    } else if (evidenceItem.media_type === 'image') {
      if (elements.videoPlayerPreview) {
        elements.videoPlayerPreview.pause();
        elements.videoPlayerPreview.style.display = 'none';
      }
      if (elements.audioViewerContainer) elements.audioViewerContainer.style.display = 'none';
      if (elements.inspectingMediaTypeIcon) elements.inspectingMediaTypeIcon.textContent = '🖼️';

      const mediaUrl = CrimeVisionAPI.getMediaUrl(evidenceItem.evidence_id);
      if (elements.imagePreview) {
        elements.imagePreview.src = mediaUrl;
      }
      if (elements.imageViewerContainer) {
        elements.imageViewerContainer.style.display = 'flex';
      }

      if (elements.videoResultsPanel) elements.videoResultsPanel.style.display = 'none';
      if (elements.audioFactsPanel) elements.audioFactsPanel.style.display = 'none';
      if (elements.imageResultsPanel) elements.imageResultsPanel.style.display = 'block';

      const res = evidenceItem.analysis;
      if (res) {
        AppState.imageStatus = 'completed';
        AppState.imageEvidence = { filename: evidenceItem.original_filename };
        AppState.imageObjectDetection = res.object_detection || null;
        AppState.imageDetections = (res.object_detection && res.object_detection.detections) || [];
        AppState.imageSceneResult = res.scene_classification || null;
        AppState.imageCaptionResult = res.image_caption || null;
        AppState.imageLlmAnalysis = res.llm_analysis || null;
        renderImageResults();
      }
    } else if (evidenceItem.media_type === 'video') {
      if (elements.imageViewerContainer) elements.imageViewerContainer.style.display = 'none';
      if (elements.audioViewerContainer) elements.audioViewerContainer.style.display = 'none';
      if (elements.inspectingMediaTypeIcon) elements.inspectingMediaTypeIcon.textContent = '🎬';

      AppState.videoPreviewUrl = CrimeVisionAPI.getMediaUrl(evidenceItem.evidence_id);
      if (elements.videoPlayerPreview) {
        elements.videoPlayerPreview.src = AppState.videoPreviewUrl;
        elements.videoPlayerPreview.style.display = 'block';
        elements.videoPlayerPreview.load();
      }

      if (elements.imageResultsPanel) elements.imageResultsPanel.style.display = 'none';
      if (elements.audioFactsPanel) elements.audioFactsPanel.style.display = 'none';
      if (elements.videoResultsPanel) elements.videoResultsPanel.style.display = 'block';

      const res = evidenceItem.analysis;
      if (res) {
        AppState.videoStatus = 'completed';
        AppState.videoEvidence = { filename: evidenceItem.original_filename };
        AppState.videoResult = res;
        AppState.videoObservations = res.frame_observations || [];
        AppState.videoEvents = (res.timeline && res.timeline.events) || [];
        AppState.videoHypothesis = res.incident_hypothesis || null;
        renderVideoResults();
      }
    }
    updateOverviewMetrics();
  }

  function clearEvidenceViewer() {
    AppState.selectedEvidenceId = null;
    if (elements.videoPlayerPreview) {
      elements.videoPlayerPreview.pause();
      elements.videoPlayerPreview.removeAttribute('src');
      elements.videoPlayerPreview.style.display = 'none';
    }
    if (elements.imageViewerContainer) elements.imageViewerContainer.style.display = 'none';
    if (elements.audioViewerContainer) elements.audioViewerContainer.style.display = 'none';
    if (elements.mediaEmptyPlaceholder) elements.mediaEmptyPlaceholder.style.display = 'flex';
    if (elements.videoResultsPanel) elements.videoResultsPanel.style.display = 'none';
    if (elements.imageResultsPanel) elements.imageResultsPanel.style.display = 'none';
    if (elements.audioFactsPanel) elements.audioFactsPanel.style.display = 'none';
    if (elements.inspectingFilename) elements.inspectingFilename.textContent = 'Awaiting Evidence Selection';
    if (elements.inspectingBadge) {
      elements.inspectingBadge.textContent = 'No Media Loaded';
      elements.inspectingBadge.className = 'badge badge-status-neutral';
    }
    if (elements.inspectingMediaTypeIcon) elements.inspectingMediaTypeIcon.textContent = '🔍';
  }

  // -------------------------------------------------------------------------
  // Audio Upload & Ingestion
  // -------------------------------------------------------------------------
  function bindUploadEvents() {
    elements.dropzone.addEventListener('click', () => elements.fileInput.click());

    elements.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dropzone.classList.add('dragover');
    });

    elements.dropzone.addEventListener('dragleave', () => {
      elements.dropzone.classList.remove('dragover');
    });

    elements.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    elements.btnAnalyze.addEventListener('click', handleAnalyze);
    elements.btnReset.addEventListener('click', handleReset);
  }

  function handleFileSelect(file) {
    const validExtensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!isValid) {
      showAlert(`Unsupported audio format. Supported: ${validExtensions.join(', ')}`, 'error');
      return;
    }

    clearAlert();
    selectedFile = file;
    elements.selectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    elements.selectedFileBox.style.display = 'flex';
    elements.btnAnalyze.disabled = false;
  }

  async function handleAnalyze() {
    if (!selectedFile || AppState.status === 'processing') return;

    AppState.status = 'processing';
    renderButtonState();
    clearAlert();

    try {
      const language = elements.languageSelect.value || null;
      const uploaded = await CrimeVisionAPI.uploadCaseEvidence(
        AppState.currentCaseId,
        selectedFile,
        { mediaType: 'audio', language, analyze: true }
      );

      const result = uploaded.analysis || (await CrimeVisionAPI.analyzeCaseEvidence(
        AppState.currentCaseId,
        (uploaded.evidence && uploaded.evidence.evidence_id) || uploaded.evidence_id,
        { language }
      ));
      const analysisData = (result && result.analysis) ? result.analysis : result;

      AppState.status = 'completed';
      AppState.evidence = {
        filename: selectedFile.name,
        duration: analysisData.recording_duration_seconds,
      };
      AppState.transcription = analysisData.transcription;
      AppState.timeline = analysisData.timeline;
      AppState.events = (analysisData.timeline && analysisData.timeline.events) || [];

      await switchCase(AppState.currentCaseId);
      showAlert('Audio evidence successfully analyzed and attached to case.', 'info');
    } catch (err) {
      AppState.status = 'error';
      AppState.error = err.message || 'An unexpected error occurred during audio processing.';
      showAlert(AppState.error, 'error');
    } finally {
      renderButtonState();
    }
  }

  function handleReset() {
    AppState.resetAudio();
    selectedFile = null;
    elements.fileInput.value = '';
    elements.selectedFileBox.style.display = 'none';
    elements.selectedFileName.textContent = '';
    elements.btnAnalyze.disabled = true;
    clearAlert();
    renderButtonState();
  }

  // -------------------------------------------------------------------------
  // Image Upload & Ingestion
  // -------------------------------------------------------------------------
  function bindImageUploadEvents() {
    elements.imageDropzone.addEventListener('click', () => elements.imageFileInput.click());

    elements.imageDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.imageDropzone.classList.add('dragover');
    });

    elements.imageDropzone.addEventListener('dragleave', () => {
      elements.imageDropzone.classList.remove('dragover');
    });

    elements.imageDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.imageDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleImageFileSelect(e.dataTransfer.files[0]);
      }
    });

    elements.imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageFileSelect(e.target.files[0]);
      }
    });

    elements.btnAnalyzeImage.addEventListener('click', handleImageAnalyze);
    elements.btnResetImage.addEventListener('click', handleImageReset);
  }

  function handleImageFileSelect(file) {
    const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!isValid) {
      showAlert(`Unsupported image format. Supported: ${validExtensions.join(', ')}`, 'error');
      return;
    }

    clearAlert();
    selectedImageFile = file;
    elements.imageSelectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    elements.imageSelectedFileBox.style.display = 'flex';
    elements.btnAnalyzeImage.disabled = false;
  }

  async function handleImageAnalyze() {
    if (!selectedImageFile || AppState.imageStatus === 'processing') return;

    AppState.imageStatus = 'processing';
    renderImageButtonState();
    clearAlert();

    try {
      const confThreshold = parseFloat(elements.confidenceThreshold.value) || 0.25;
      const uploaded = await CrimeVisionAPI.uploadCaseEvidence(
        AppState.currentCaseId,
        selectedImageFile,
        { mediaType: 'image', confidenceThreshold: confThreshold, analyze: true }
      );

      const result = uploaded.analysis || (await CrimeVisionAPI.analyzeCaseEvidence(
        AppState.currentCaseId,
        (uploaded.evidence && uploaded.evidence.evidence_id) || uploaded.evidence_id,
        { confidence_threshold: confThreshold }
      ));
      const analysisData = (result && result.analysis) ? result.analysis : result;

      AppState.imageStatus = 'completed';
      AppState.imageEvidence = { filename: selectedImageFile.name };
      AppState.imageObjectDetection = analysisData.object_detection || null;
      AppState.imageDetections = (analysisData.object_detection && analysisData.object_detection.detections) || [];
      AppState.imageSceneResult = analysisData.scene_classification || null;
      AppState.imageCaptionResult = analysisData.image_caption || null;
      AppState.imageLlmAnalysis = analysisData.llm_analysis || null;

      await switchCase(AppState.currentCaseId);
      showAlert('Image evidence analyzed and recorded into case investigation.', 'info');
    } catch (err) {
      AppState.imageStatus = 'error';
      AppState.imageError = err.message || 'An unexpected error occurred during image processing.';
      showAlert(AppState.imageError, 'error');
    } finally {
      renderImageButtonState();
    }
  }

  function handleImageReset() {
    AppState.resetImage();
    selectedImageFile = null;
    elements.imageFileInput.value = '';
    elements.imageSelectedFileBox.style.display = 'none';
    elements.imageSelectedFileName.textContent = '';
    elements.btnAnalyzeImage.disabled = true;
    clearAlert();
    renderImageButtonState();
  }

  // -------------------------------------------------------------------------
  // Video Upload & Ingestion (Epic 4)
  // -------------------------------------------------------------------------
  function bindVideoUploadEvents() {
    elements.videoDropzone.addEventListener('click', () => elements.videoFileInput.click());

    elements.videoDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.videoDropzone.classList.add('dragover');
    });

    elements.videoDropzone.addEventListener('dragleave', () => {
      elements.videoDropzone.classList.remove('dragover');
    });

    elements.videoDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.videoDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleVideoFileSelect(e.dataTransfer.files[0]);
      }
    });

    elements.videoFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleVideoFileSelect(e.target.files[0]);
      }
    });

    elements.btnAnalyzeVideo.addEventListener('click', handleVideoAnalyze);
    elements.btnResetVideo.addEventListener('click', handleVideoReset);
  }

  function handleVideoFileSelect(file) {
    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!isValid) {
      showAlert(`Unsupported video format. Supported: ${validExtensions.join(', ')}`, 'error');
      return;
    }

    clearAlert();
    selectedVideoFile = file;
    elements.videoSelectedFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    elements.videoSelectedFileBox.style.display = 'flex';
    elements.btnAnalyzeVideo.disabled = false;
  }

  async function handleVideoAnalyze() {
    if (!selectedVideoFile || AppState.videoStatus === 'processing') return;

    AppState.videoStatus = 'processing';
    renderVideoButtonState();
    clearAlert();

    try {
      const confThreshold = parseFloat(elements.videoConfidenceThreshold.value) || 0.25;
      const uploaded = await CrimeVisionAPI.uploadCaseEvidence(
        AppState.currentCaseId,
        selectedVideoFile,
        { mediaType: 'video', confidenceThreshold: confThreshold, analyze: true }
      );

      const result = uploaded.analysis || (await CrimeVisionAPI.analyzeCaseEvidence(
        AppState.currentCaseId,
        (uploaded.evidence && uploaded.evidence.evidence_id) || uploaded.evidence_id,
        { confidence_threshold: confThreshold }
      ));
      const analysisData = (result && result.analysis) ? result.analysis : result;

      AppState.videoStatus = 'completed';
      AppState.videoEvidence = { filename: selectedVideoFile.name };
      AppState.videoResult = analysisData;
      AppState.videoObservations = analysisData.frame_observations || [];
      AppState.videoEvents = (analysisData.timeline && analysisData.timeline.events) || [];
      AppState.videoHypothesis = analysisData.incident_hypothesis || null;

      await switchCase(AppState.currentCaseId);
      showAlert('Video surveillance evidence analyzed and attached to investigation.', 'info');
    } catch (err) {
      AppState.videoStatus = 'error';
      AppState.videoError = err.message || 'An unexpected error occurred during video processing.';
      showAlert(AppState.videoError, 'error');
    } finally {
      renderVideoButtonState();
    }
  }

  function handleVideoReset() {
    AppState.resetVideo();
    selectedVideoFile = null;
    elements.videoFileInput.value = '';
    elements.videoSelectedFileBox.style.display = 'none';
    elements.videoSelectedFileName.textContent = '';
    elements.btnAnalyzeVideo.disabled = true;
    clearAlert();
    renderVideoButtonState();
  }

  // -------------------------------------------------------------------------
  // Ingestion Tab Switching (Changes ONLY upload form, NOT inspected evidence)
  // -------------------------------------------------------------------------
  function bindTabEvents() {
    if (elements.tabAudio) {
      elements.tabAudio.addEventListener('click', () => switchTab('audio'));
    }

    if (elements.tabImage) {
      elements.tabImage.addEventListener('click', () => switchTab('image'));
    }

    if (elements.tabVideo) {
      elements.tabVideo.addEventListener('click', () => switchTab('video'));
    }
  }

  function switchTab(tab) {
    AppState.activeTab = tab;
    clearAlert();

    [elements.tabAudio, elements.tabImage, elements.tabVideo].forEach((btn) => {
      if (btn) {
        btn.classList.remove('active-tab');
        btn.setAttribute('aria-selected', 'false');
      }
    });

    if (tab === 'audio') {
      if (elements.tabAudio) {
        elements.tabAudio.classList.add('active-tab');
        elements.tabAudio.setAttribute('aria-selected', 'true');
      }
      if (elements.audioUploadSection) elements.audioUploadSection.style.display = 'block';
      if (elements.imageUploadSection) elements.imageUploadSection.style.display = 'none';
      if (elements.videoUploadSection) elements.videoUploadSection.style.display = 'none';
      if (elements.ingestionPanelIcon) elements.ingestionPanelIcon.textContent = '🎙️';
    } else if (tab === 'image') {
      if (elements.tabImage) {
        elements.tabImage.classList.add('active-tab');
        elements.tabImage.setAttribute('aria-selected', 'true');
      }
      if (elements.audioUploadSection) elements.audioUploadSection.style.display = 'none';
      if (elements.imageUploadSection) elements.imageUploadSection.style.display = 'block';
      if (elements.videoUploadSection) elements.videoUploadSection.style.display = 'none';
      if (elements.ingestionPanelIcon) elements.ingestionPanelIcon.textContent = '🖼️';
    } else if (tab === 'video') {
      if (elements.tabVideo) {
        elements.tabVideo.classList.add('active-tab');
        elements.tabVideo.setAttribute('aria-selected', 'true');
      }
      if (elements.audioUploadSection) elements.audioUploadSection.style.display = 'none';
      if (elements.imageUploadSection) elements.imageUploadSection.style.display = 'none';
      if (elements.videoUploadSection) elements.videoUploadSection.style.display = 'block';
      if (elements.ingestionPanelIcon) elements.ingestionPanelIcon.textContent = '🎬';
    }
  }

  // -------------------------------------------------------------------------
  // Investigator Grounded Q&A Chat (Epic 4 Section 7)
  // -------------------------------------------------------------------------
  function bindChatEvents() {
    if (elements.btnChatSend) {
      elements.btnChatSend.addEventListener('click', handleChatSubmit);
    }

    if (elements.chatInput) {
      elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          handleChatSubmit();
        }
      });
    }
  }

  function formatHypothesisMarkdown(result) {
    const hypotheses = result.hypotheses || [];
    const sufficiency = (result.evidence_sufficiency || 'unknown').toUpperCase();
    let md = `### 🔍 Grounded Investigative Hypotheses\n\n`;
    md += `**Evidence Sufficiency:** \`${sufficiency}\`\n\n`;

    if (hypotheses.length === 0) {
      md += `*Based on recorded case findings, available evidence is insufficient to formulate grounded hypotheses.*\n\n`;
    } else {
      hypotheses.forEach((h, idx) => {
        const plausibility = (h.plausibility || 'unrated').toUpperCase();
        md += `#### ${idx + 1}. ${h.title || 'Hypothesis'} *(Plausibility: ${plausibility})*\n`;
        if (h.narrative) {
          md += `${h.narrative}\n\n`;
        }

        if (h.supporting_evidence && h.supporting_evidence.length > 0) {
          md += `**Supporting Evidence:**\n`;
          h.supporting_evidence.forEach((item) => {
            md += `- ${item}\n`;
          });
          md += `\n`;
        }

        if (h.contradicting_evidence && h.contradicting_evidence.length > 0) {
          md += `**Contradicting / Missing Evidence:**\n`;
          h.contradicting_evidence.forEach((item) => {
            md += `- ${item}\n`;
          });
          md += `\n`;
        }

        if (h.assumptions && h.assumptions.length > 0) {
          md += `**Key Assumptions:**\n`;
          h.assumptions.forEach((item) => {
            md += `- ${item}\n`;
          });
          md += `\n`;
        }

        if (h.uncertainty_factors && h.uncertainty_factors.length > 0) {
          md += `**Uncertainty Factors:**\n`;
          h.uncertainty_factors.forEach((item) => {
            md += `- ${item}\n`;
          });
          md += `\n`;
        }

        if (idx < hypotheses.length - 1) {
          md += `---\n\n`;
        }
      });
    }

    if (result.overall_limitations) {
      md += `> **Overall Forensic Limitations:** ${result.overall_limitations}\n`;
    }

    return md;
  }

  async function handleChatSubmit() {
    const question = elements.chatInput.value.trim();
    if (!question) return;

    elements.chatInput.value = '';

    appendChatMessage('user', question);
    const typingMsg = appendChatMessage('ai', 'Reviewing case findings...');
    elements.btnChatSend.disabled = true;

    try {
      const result = await CrimeVisionAPI.chat(AppState.currentCaseId, question);
      typingMsg.remove();
      if (result.query_type === 'hypothesis') {
        const md = formatHypothesisMarkdown(result);
        appendChatMessage('ai', md, result.note || 'AI-generated investigative hypotheses grounded in case records.');
      } else {
        appendChatMessage('ai', result.answer || 'No findings reported.', result.note);
      }
    } catch (err) {
      typingMsg.remove();
      appendChatMessage('ai', `Unable to answer: ${err.message}`);
    } finally {
      elements.btnChatSend.disabled = false;
      if (elements.chatMessagesContainer) {
        elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Safe Markdown Parser & Renderer for Investigator Q&A
  // -------------------------------------------------------------------------
  const MarkdownRenderer = {
    parseTableCells(line) {
      let trimmed = line.trim();
      if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
      if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1);
      return trimmed.split('|').map((c) => c.trim());
    },

    isTableSeparator(line) {
      if (!line || !line.includes('|')) return false;
      const cells = this.parseTableCells(line);
      if (cells.length < 1) return false;
      return cells.every((c) => /^:?-{2,}:?$/.test(c));
    },

    getTableAlignments(delimiterLine) {
      const cells = this.parseTableCells(delimiterLine);
      return cells.map((c) => {
        const left = c.startsWith(':');
        const right = c.endsWith(':');
        if (left && right) return 'center';
        if (right) return 'right';
        if (left) return 'left';
        return '';
      });
    },

    renderInline(text, container) {
      if (!text) return;
      const inlineRegex = /(`[^`]+`|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
      let lastIndex = 0;
      let match;

      while ((match = inlineRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
          container.appendChild(document.createTextNode(text.substring(lastIndex, match.index)));
        }
        const token = match[0];
        if (token.startsWith('`') && token.endsWith('`')) {
          const code = document.createElement('code');
          code.textContent = token.slice(1, -1);
          container.appendChild(code);
        } else if (token.startsWith('***') && token.endsWith('***')) {
          const strong = document.createElement('strong');
          const em = document.createElement('em');
          em.textContent = token.slice(3, -3);
          strong.appendChild(em);
          container.appendChild(strong);
        } else if ((token.startsWith('**') && token.endsWith('**')) || (token.startsWith('__') && token.endsWith('__'))) {
          const strong = document.createElement('strong');
          strong.textContent = token.slice(2, -2);
          container.appendChild(strong);
        } else if ((token.startsWith('*') && token.endsWith('*')) || (token.startsWith('_') && token.endsWith('_'))) {
          const em = document.createElement('em');
          em.textContent = token.slice(1, -1);
          container.appendChild(em);
        } else if (token.startsWith('[') && match[2] && match[3]) {
          const a = document.createElement('a');
          a.textContent = match[2];
          a.href = match[3];
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          container.appendChild(a);
        }
        lastIndex = inlineRegex.lastIndex;
      }

      if (lastIndex < text.length) {
        container.appendChild(document.createTextNode(text.substring(lastIndex)));
      }
    },

    renderParagraphLines(lines, container) {
      lines.forEach((line, idx) => {
        if (idx > 0) container.appendChild(document.createElement('br'));
        this.renderInline(line, container);
      });
    },

    render(markdown, container) {
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }

      if (!markdown) return;

      const lines = markdown.replace(/\r\n/g, '\n').split('\n');
      let i = 0;

      while (i < lines.length) {
        const rawLine = lines[i];
        const line = rawLine.trim();

        if (!line) {
          i++;
          continue;
        }

        // 1. Headings: #, ##, ###, ####
        const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
          const level = headingMatch[1].length;
          const h = document.createElement(`h${Math.min(level + 1, 6)}`);
          h.className = `chat-heading chat-h${level}`;
          this.renderInline(headingMatch[2], h);
          container.appendChild(h);
          i++;
          continue;
        }

        // 2. Horizontal rule: --- or ***
        if (/^(\*{3,}|-{3,}|_{3,})$/.test(line)) {
          const hr = document.createElement('hr');
          hr.className = 'chat-divider';
          container.appendChild(hr);
          i++;
          continue;
        }

        // 3. Tables
        if (line.includes('|') && i + 1 < lines.length && this.isTableSeparator(lines[i + 1])) {
          const headerCells = this.parseTableCells(line);
          const alignments = this.getTableAlignments(lines[i + 1]);
          i += 2;

          const table = document.createElement('table');
          table.className = 'chat-table';
          const thead = document.createElement('thead');
          const headerRow = document.createElement('tr');

          headerCells.forEach((cText, colIdx) => {
            const th = document.createElement('th');
            if (alignments[colIdx]) th.style.textAlign = alignments[colIdx];
            this.renderInline(cText, th);
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);
          table.appendChild(thead);

          const tbody = document.createElement('tbody');
          while (i < lines.length && lines[i].trim().includes('|')) {
            const rowCells = this.parseTableCells(lines[i]);
            const tr = document.createElement('tr');
            rowCells.forEach((cellText, colIdx) => {
              const td = document.createElement('td');
              if (alignments[colIdx]) td.style.textAlign = alignments[colIdx];
              this.renderInline(cellText, td);
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
            i++;
          }
          table.appendChild(tbody);
          // Wrap table in a scrollable container to prevent bubble overflow
          const tableWrap = document.createElement('div');
          tableWrap.className = 'chat-table-wrap';
          tableWrap.appendChild(table);
          container.appendChild(tableWrap);
          continue;
        }

        // 4. Unordered list: - or *
        if (/^\s*[-*+]\s+/.test(line)) {
          const ul = document.createElement('ul');
          ul.className = 'chat-list chat-ul';
          while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
            const liText = lines[i].replace(/^\s*[-*+]\s+/, '');
            const li = document.createElement('li');
            this.renderInline(liText, li);
            ul.appendChild(li);
            i++;
          }
          container.appendChild(ul);
          continue;
        }

        // 5. Ordered list: 1. item
        if (/^\s*\d+\.\s+/.test(line)) {
          const ol = document.createElement('ol');
          ol.className = 'chat-list chat-ol';
          while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
            const liText = lines[i].replace(/^\s*\d+\.\s+/, '');
            const li = document.createElement('li');
            this.renderInline(liText, li);
            ol.appendChild(li);
            i++;
          }
          container.appendChild(ol);
          continue;
        }

        // 6. Blockquote: > quote
        if (/^\s*>\s*/.test(line)) {
          const bq = document.createElement('blockquote');
          bq.className = 'chat-blockquote';
          const bqLines = [];
          while (i < lines.length && /^\s*>\s*/.test(lines[i])) {
            bqLines.push(lines[i].replace(/^\s*>\s*/, ''));
            i++;
          }
          const p = document.createElement('p');
          this.renderParagraphLines(bqLines, p);
          bq.appendChild(p);
          container.appendChild(bq);
          continue;
        }

        // 7. Regular paragraph
        const pLines = [];
        while (
          i < lines.length &&
          lines[i].trim() &&
          !/^#{1,6}\s+/.test(lines[i].trim()) &&
          !(lines[i].trim().includes('|') && i + 1 < lines.length && this.isTableSeparator(lines[i + 1])) &&
          !/^\s*[-*+]\s+/.test(lines[i]) &&
          !/^\s*\d+\.\s+/.test(lines[i]) &&
          !/^\s*>\s*/.test(lines[i])
        ) {
          pLines.push(lines[i]);
          i++;
        }

        if (pLines.length > 0) {
          const p = document.createElement('p');
          this.renderParagraphLines(pLines, p);
          container.appendChild(p);
        }
      }
    },
  };

  window.CrimeVisionMarkdown = MarkdownRenderer;

  function appendChatMessage(sender, text, citation = null) {
    const div = document.createElement('div');
    div.className = `chat-msg chat-msg-${sender} chat-message ${sender === 'user' ? 'user' : 'assistant'}`;

    if (sender === 'ai' || sender === 'assistant') {
      const contentDiv = document.createElement('div');
      contentDiv.className = 'chat-msg-content';
      MarkdownRenderer.render(text, contentDiv);
      div.appendChild(contentDiv);
    } else {
      const p = document.createElement('p');
      p.textContent = text;
      div.appendChild(p);
    }

    if (citation) {
      const cite = document.createElement('span');
      cite.className = 'chat-citation';
      cite.textContent = citation;
      div.appendChild(cite);
    }

    elements.chatMessagesContainer.appendChild(div);
    elements.chatMessagesContainer.scrollTop = elements.chatMessagesContainer.scrollHeight;
    return div;
  }

  // -------------------------------------------------------------------------
  // Export Logic
  // -------------------------------------------------------------------------
  function bindExportEvents() {
    if (elements.btnExportJson) {
      elements.btnExportJson.addEventListener('click', async () => {
        if (!AppState.currentCase) {
          showAlert('No case data available to export.', 'error');
          return;
        }

        let exportPayload = null;
        try {
          const response = await fetch(`/api/cases/${encodeURIComponent(AppState.currentCaseId)}/export`);
          if (response.ok) {
            const data = await response.json();
            exportPayload = data.export;
          }
        } catch (_) {}

        if (!exportPayload) {
          exportPayload = {
            case_id: AppState.currentCaseId,
            title: AppState.currentCase.title,
            description: AppState.currentCase.description,
            created_at: AppState.currentCase.created_at,
            updated_at: AppState.currentCase.updated_at,
            exported_at: new Date().toISOString(),
            evidence_count: (AppState.currentCase.evidence || []).length,
            evidence: AppState.currentCase.evidence || [],
            timeline_events: AppState.currentCase.timeline_events || [],
            fusion: AppState.fusion || AppState.currentCase.fusion || null,
            forensic_metadata: {
              system: 'CrimeVisionAI Multimodal Forensic Analysis Platform',
              audit_notice: (
                'Automated model predictions (speech recognition, object detection, ' +
                'scene classification, automated cross-referencing, and AI hypotheses) ' +
                'are probabilistic interpretations and do not constitute verified facts.'
              ),
            },
          };
        }

        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportPayload, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute('href', dataStr);
        downloadAnchor.setAttribute('download', `crimevision_${AppState.currentCaseId}_case_export.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
        showAlert('Forensic case timeline and metadata exported successfully.', 'info');
      });
    }

    if (elements.btnExportPdf) {
      elements.btnExportPdf.addEventListener('click', async () => {
        if (!AppState.currentCaseId) {
          showAlert('No case selected to export.', 'error');
          return;
        }

        elements.btnExportPdf.disabled = true;
        const originalText = elements.btnExportPdf.textContent;
        elements.btnExportPdf.textContent = '📑 Generating...';
        showAlert('Compiling official A4 forensic PDF report...', 'info');

        try {
          const res = await fetch(`/api/cases/${encodeURIComponent(AppState.currentCaseId)}/export/pdf`);
          if (!res.ok) {
            let errText = `HTTP ${res.status}`;
            try {
              const errData = await res.json();
              errText = errData.message || errData.error || errText;
            } catch (_) {}
            throw new Error(errText);
          }

          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = `crimevision_${AppState.currentCaseId}_investigative_report.pdf`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          a.remove();

          showAlert('Forensic PDF investigative report downloaded successfully.', 'info');
        } catch (err) {
          showAlert(`Failed to generate PDF report: ${err.message}`, 'error');
        } finally {
          elements.btnExportPdf.disabled = false;
          elements.btnExportPdf.textContent = originalText;
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // Audio State & Segments Rendering
  // -------------------------------------------------------------------------
  function renderState() {
    if (AppState.status === 'processing') {
      elements.factsContainer.style.display = 'none';
      elements.transcriptDisplay.textContent = 'Analyzing speech audio and acoustic segments...';
      while (elements.segmentsList.firstChild) {
        elements.segmentsList.removeChild(elements.segmentsList.firstChild);
      }
      return;
    }

    if (AppState.status === 'completed' && AppState.evidence) {
      elements.factsContainer.style.display = 'block';
      elements.factFile.textContent = AppState.evidence.filename || '--';
      elements.factDuration.textContent = AppState.evidence.duration !== null && AppState.evidence.duration !== undefined
        ? `${AppState.evidence.duration.toFixed(1)}s`
        : '--';

      if (AppState.transcription) {
        const rawLang = AppState.transcription.detected_language || AppState.transcription.language || '';
        elements.factLanguage.textContent = rawLang === 'en' ? 'English' : (rawLang || 'English');
        elements.factQuality.textContent = AppState.transcription.timestamp_quality || 'word_level';
        const transcriptText = AppState.transcription.transcript !== undefined ? AppState.transcription.transcript : AppState.transcription.text;
        elements.transcriptDisplay.textContent = (transcriptText && transcriptText.trim()) ? transcriptText : 'No spoken words detected.';
      }

      renderSegments((AppState.transcription && AppState.transcription.segments) || []);
    } else {
      elements.factsContainer.style.display = 'none';
      elements.transcriptDisplay.textContent = 'Awaiting audio analysis...';
      while (elements.segmentsList.firstChild) {
        elements.segmentsList.removeChild(elements.segmentsList.firstChild);
      }
    }
  }

  function renderSegments(segments) {
    while (elements.segmentsList.firstChild) {
      elements.segmentsList.removeChild(elements.segmentsList.firstChild);
    }

    segments.forEach((seg) => {
      const item = document.createElement('div');
      item.className = 'segment-item';

      const timeSpan = document.createElement('span');
      timeSpan.className = 'segment-time';
      const s = seg.start !== null && seg.start !== undefined ? `${seg.start.toFixed(1)}s` : '?';
      const e = seg.end !== null && seg.end !== undefined ? `${seg.end.toFixed(1)}s` : '?';
      timeSpan.textContent = `[${s} - ${e}]`;

      const textSpan = document.createElement('span');
      textSpan.className = 'segment-text';
      textSpan.textContent = seg.text;

      item.appendChild(timeSpan);
      item.appendChild(textSpan);
      elements.segmentsList.appendChild(item);
    });
  }

  // -------------------------------------------------------------------------
  // Consolidated Case Timeline Rendering (Audio + Video)
  // -------------------------------------------------------------------------
  function renderCaseTimeline(events) {
    while (elements.timelineEventsList.firstChild) {
      elements.timelineEventsList.removeChild(elements.timelineEventsList.firstChild);
    }

    const count = events.length;
    elements.timelineCountBadge.textContent = `${count} Event${count !== 1 ? 's' : ''}`;

    if (count === 0) {
      elements.timelineEmptyNotice.style.display = 'block';
      elements.timelineEventsList.style.display = 'none';
      elements.btnExportJson.disabled = true;
      if (elements.btnExportPdf) {
        elements.btnExportPdf.disabled = !AppState.currentCaseId;
      }
      return;
    }

    elements.timelineEmptyNotice.style.display = 'none';
    elements.timelineEventsList.style.display = 'flex';
    elements.btnExportJson.disabled = false;
    if (elements.btnExportPdf) {
      elements.btnExportPdf.disabled = false;
    }

    events.forEach((ev) => {
      const card = document.createElement('div');
      card.className = 'timeline-event-card';

      const header = document.createElement('div');
      header.className = 'event-header';

      const badgeWrap = document.createElement('div');
      badgeWrap.style.display = 'flex';
      badgeWrap.style.gap = '0.35rem';
      badgeWrap.style.alignItems = 'center';

      const typeBadge = document.createElement('span');
      typeBadge.className = 'event-type-badge';
      typeBadge.textContent = `${ev.event_id || 'EVT'} • ${(ev.event_type || 'unclassified').replace(/_/g, ' ')}`;
      badgeWrap.appendChild(typeBadge);

      const scoreVal = typeof ev.certainty_score === 'number'
        ? ev.certainty_score
        : (typeof ev.evidence_consistency_score === 'number' ? ev.evidence_consistency_score : null);
      if (scoreVal !== null) {
        const certBadge = document.createElement('span');
        certBadge.style.fontSize = '0.7rem';
        certBadge.style.color = '#38bdf8';
        certBadge.style.background = 'rgba(56, 189, 248, 0.15)';
        certBadge.style.padding = '0.1rem 0.35rem';
        certBadge.style.borderRadius = '3px';
        certBadge.textContent = `Score: ${scoreVal.toFixed(2)}`;
        badgeWrap.appendChild(certBadge);
      }

      if (ev.anchors && ev.anchors.length > 0) {
        const anchorBadge = document.createElement('span');
        anchorBadge.style.fontSize = '0.7rem';
        anchorBadge.style.color = '#c084fc';
        anchorBadge.style.background = 'rgba(192, 132, 252, 0.15)';
        anchorBadge.style.padding = '0.1rem 0.35rem';
        anchorBadge.style.borderRadius = '3px';
        anchorBadge.textContent = '⚓ Anchor';
        badgeWrap.appendChild(anchorBadge);
      }

      const timeWrap = document.createElement('div');
      timeWrap.style.display = 'flex';
      timeWrap.style.alignItems = 'center';
      timeWrap.style.gap = '0.35rem';

      const timeChip = document.createElement('span');
      timeChip.className = 'event-time-chip';
      const rawStart = ev.start_raw !== null && ev.start_raw !== undefined
        ? ev.start_raw
        : (ev.timestamp_normalized !== null && ev.timestamp_normalized !== undefined ? ev.timestamp_normalized : null);
      const rawEnd = ev.end_raw !== null && ev.end_raw !== undefined
        ? ev.end_raw
        : (ev.end_timestamp_normalized !== null && ev.end_timestamp_normalized !== undefined ? ev.end_timestamp_normalized : null);

      const s = typeof rawStart === 'number' ? `${rawStart.toFixed(1)}s` : 'Unknown';
      const e = typeof rawEnd === 'number' ? `${rawEnd.toFixed(1)}s` : 'Unknown';
      timeChip.textContent = `⏱ ${s} - ${e}`;
      timeWrap.appendChild(timeChip);

      const resolvedMediaType = (ev.media_type || ev.modality || '').toLowerCase();
      // Add Seek Video button if it's a video event and timestamp exists
      if (resolvedMediaType === 'video' && typeof rawStart === 'number') {
        const btnSeek = document.createElement('button');
        btnSeek.className = 'btn-seek';
        btnSeek.dataset.timestamp = String(rawStart);
        btnSeek.textContent = '▶ Seek';
        btnSeek.title = `Seek video to ${rawStart.toFixed(1)}s`;
        btnSeek.addEventListener('click', (e) => {
          e.stopPropagation();
          if (elements.videoPlayerPreview && elements.videoPlayerPreview.src) {
            elements.videoPlayerPreview.currentTime = rawStart;
            elements.videoPlayerPreview.play().catch(() => {});
            elements.videoPlayerPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        });
        timeWrap.appendChild(btnSeek);
      }

      header.appendChild(badgeWrap);
      header.appendChild(timeWrap);
      card.appendChild(header);

      const desc = document.createElement('p');
      desc.className = 'event-desc';
      desc.textContent = ev.description || '';
      card.appendChild(desc);

      const meta = document.createElement('div');
      meta.className = 'event-meta';
      meta.style.display = 'flex';
      meta.style.justifyContent = 'space-between';
      meta.style.fontSize = '0.7rem';
      meta.style.color = 'var(--text-dim)';

      const resolvedFile = ev.source_filename || ev.original_filename || 'unknown';
      const modSpan = document.createElement('span');
      modSpan.textContent = `Source: ${resolvedFile} (${(ev.media_type || ev.modality || 'file').toUpperCase()})`;
      meta.appendChild(modSpan);

      if (ev.temporal_anchor_id) {
        const ancSpan = document.createElement('span');
        ancSpan.textContent = `Anchor: ${ev.temporal_anchor_id}`;
        meta.appendChild(ancSpan);
      }

      card.appendChild(meta);
      elements.timelineEventsList.appendChild(card);
    });
  }

  // -------------------------------------------------------------------------
  // Video Analysis Results Rendering (Epic 4)
  // -------------------------------------------------------------------------
  function renderVideoResults() {
    const res = AppState.videoResult;
    if (!res) return;

    elements.videoFramesCountBadge.textContent = `${res.analyzed_frame_count || 0} Frames`;

    elements.videoFactFile.textContent = res.video_file || (AppState.videoEvidence ? AppState.videoEvidence.filename : '--');
    elements.videoFactDuration.textContent = res.duration_seconds !== null && res.duration_seconds !== undefined
      ? `${res.duration_seconds.toFixed(1)}s`
      : '--';

    const span = res.analyzed_interval;
    elements.videoFactSpan.textContent = span
      ? `[${span.start_seconds.toFixed(1)}s - ${span.end_seconds.toFixed(1)}s]`
      : '--';
    elements.videoFactStatus.textContent = (res.status || 'complete').toUpperCase();

    // Incident Hypothesis
    const hyp = res.incident_hypothesis || {};
    elements.videoHypType.textContent = (hyp.type || 'insufficient_evidence').replace(/_/g, ' ').toUpperCase();
    elements.videoHypDesc.textContent = hyp.description || 'Awaiting video analysis...';

    // Inferred Visual Anchor Events
    renderVideoEvents(AppState.videoEvents);

    // Sampled Frame Observations
    renderVideoObservations(AppState.videoObservations);
  }

  function renderVideoEvents(events) {
    while (elements.videoEventsList.firstChild) {
      elements.videoEventsList.removeChild(elements.videoEventsList.firstChild);
    }

    if (!events || events.length === 0) {
      const empty = document.createElement('p');
      empty.style.color = 'var(--text-dim)';
      empty.style.fontSize = '0.8rem';
      empty.textContent = 'No anchor action events inferred for this video.';
      elements.videoEventsList.appendChild(empty);
      return;
    }

    events.forEach((ev) => {
      const card = document.createElement('div');
      card.className = 'timeline-event-card';

      const header = document.createElement('div');
      header.className = 'event-header';

      const typeBadge = document.createElement('span');
      typeBadge.className = 'event-type-badge';
      typeBadge.textContent = `${ev.event_id || 'EVT'} • ${(ev.event_type || 'action').replace(/_/g, ' ')}`;

      const timeWrap = document.createElement('div');
      timeWrap.style.display = 'flex';
      timeWrap.style.alignItems = 'center';
      timeWrap.style.gap = '0.35rem';

      const timeChip = document.createElement('span');
      timeChip.className = 'event-time-chip';
      timeChip.textContent = `⏱ ${(ev.timestamp_seconds || 0).toFixed(1)}s`;

      const btnSeek = document.createElement('button');
      btnSeek.className = 'btn-seek';
      btnSeek.dataset.timestamp = String(ev.timestamp_seconds || 0);
      btnSeek.textContent = '▶ Seek';
      btnSeek.addEventListener('click', (e) => {
        e.stopPropagation();
        if (elements.videoPlayerPreview && elements.videoPlayerPreview.src) {
          elements.videoPlayerPreview.currentTime = ev.timestamp_seconds || 0;
          elements.videoPlayerPreview.play().catch(() => {});
          elements.videoPlayerPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      });

      timeWrap.appendChild(timeChip);
      timeWrap.appendChild(btnSeek);
      header.appendChild(typeBadge);
      header.appendChild(timeWrap);
      card.appendChild(header);

      const desc = document.createElement('p');
      desc.className = 'event-desc';
      desc.textContent = ev.description || '';
      card.appendChild(desc);

      elements.videoEventsList.appendChild(card);
    });
  }

  function renderVideoObservations(frames) {
    while (elements.videoFramesList.firstChild) {
      elements.videoFramesList.removeChild(elements.videoFramesList.firstChild);
    }

    if (!frames || frames.length === 0) {
      const empty = document.createElement('p');
      empty.style.color = 'var(--text-dim)';
      empty.style.fontSize = '0.8rem';
      empty.textContent = 'No per-frame observations available.';
      elements.videoFramesList.appendChild(empty);
      return;
    }

    frames.forEach((f) => {
      const item = document.createElement('div');
      item.className = 'segment-item';
      item.style.cursor = 'pointer';
      item.title = 'Click to seek video preview to this frame';
      item.dataset.timestamp = String(f.timestamp_seconds || 0);
      item.addEventListener('click', () => {
        if (elements.videoPlayerPreview && elements.videoPlayerPreview.src) {
          elements.videoPlayerPreview.currentTime = f.timestamp_seconds || 0;
          elements.videoPlayerPreview.play().catch(() => {});
          elements.videoPlayerPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      });

      const timeSpan = document.createElement('span');
      timeSpan.className = 'segment-time';
      timeSpan.textContent = `[${f.timestamp_seconds.toFixed(1)}s] Frame #${f.frame_index}`;

      const textSpan = document.createElement('span');
      textSpan.className = 'segment-text';
      textSpan.textContent = f.caption || (f.detected_objects && f.detected_objects.join(', ')) || 'Frame sampled';

      item.appendChild(timeSpan);
      item.appendChild(textSpan);
      elements.videoFramesList.appendChild(item);
    });
  }

  // -------------------------------------------------------------------------
  // Image Analysis Results Rendering
  // -------------------------------------------------------------------------
  function renderImageResults() {
    const od = AppState.imageObjectDetection;
    const detections = AppState.imageDetections;
    const scene = AppState.imageSceneResult;
    const caption = AppState.imageCaptionResult;
    const llm = AppState.imageLlmAnalysis;

    elements.imageFactsContainer.style.display = 'block';
    elements.imageFactFile.textContent = AppState.imageEvidence ? AppState.imageEvidence.filename : '--';
    elements.imageFactScene.textContent = (scene && scene.top_scene) ? scene.top_scene : '--';
    elements.imageFactPersons.textContent = od ? String(od.person_detection_count || 0) : '--';

    const detCount = detections.length;
    elements.imageDetectionCountBadge.textContent = `${detCount} Detection${detCount !== 1 ? 's' : ''}`;

    if (caption && caption.caption) {
      elements.imageCaptionGroup.style.display = 'block';
      elements.imageCaptionDisplay.textContent = caption.caption;
    } else {
      elements.imageCaptionGroup.style.display = 'none';
    }

    if (detections.length > 0) {
      elements.imageDetectionsGroup.style.display = 'block';
      elements.imageCountNote.textContent = od && od.count_note ? od.count_note : '';
      renderDetections(detections);
    } else {
      elements.imageDetectionsGroup.style.display = 'none';
    }

    if (scene && scene.candidates && scene.candidates.length > 0) {
      elements.imageSceneGroup.style.display = 'block';
      renderSceneCandidates(scene.candidates);
    } else {
      elements.imageSceneGroup.style.display = 'none';
    }

    if (llm && llm.analysis) {
      elements.imageLlmGroup.style.display = 'block';
      elements.imageLlmDisplay.textContent = llm.analysis;
    } else {
      elements.imageLlmGroup.style.display = 'none';
    }
  }

  function renderDetections(detections) {
    while (elements.imageDetectionsList.firstChild) {
      elements.imageDetectionsList.removeChild(elements.imageDetectionsList.firstChild);
    }

    detections.forEach((det) => {
      const item = document.createElement('div');
      item.className = 'detection-item';

      const header = document.createElement('div');
      header.className = 'detection-item-header';
      header.style.display = 'flex';
      header.style.justifyContent = 'space-between';
      header.style.width = '100%';

      const labelSpan = document.createElement('span');
      labelSpan.className = 'detection-label';
      labelSpan.style.fontWeight = '600';
      labelSpan.textContent = det.label || 'unknown';

      const confSpan = document.createElement('span');
      confSpan.className = 'detection-confidence';
      confSpan.style.color = 'var(--text-muted)';
      const confPct = (typeof det.confidence === 'number') ? (det.confidence * 100).toFixed(1) + '%' : '--';
      confSpan.textContent = confPct;

      header.appendChild(labelSpan);
      header.appendChild(confSpan);
      item.appendChild(header);

      if (det.bbox_xyxy && det.bbox_xyxy.length === 4) {
        const bbox = document.createElement('div');
        bbox.className = 'detection-bbox';
        bbox.style.fontSize = '0.68rem';
        bbox.style.color = 'var(--text-dim)';
        bbox.style.fontFamily = 'var(--font-mono)';
        const coords = det.bbox_xyxy.map((v) => (typeof v === 'number' ? v.toFixed(0) : v));
        bbox.textContent = `bbox: [${coords.join(', ')}]`;
        item.appendChild(bbox);
      }

      elements.imageDetectionsList.appendChild(item);
    });
  }

  function renderSceneCandidates(candidates) {
    while (elements.imageSceneList.firstChild) {
      elements.imageSceneList.removeChild(elements.imageSceneList.firstChild);
    }

    candidates.forEach((c) => {
      const item = document.createElement('div');
      item.className = 'scene-candidate-item';
      item.style.display = 'flex';
      item.style.justifyContent = 'space-between';
      item.style.padding = '0.25rem 0.5rem';
      item.style.fontSize = '0.75rem';

      const label = document.createElement('span');
      label.className = 'scene-label';
      label.textContent = c.scene || '--';

      const score = document.createElement('span');
      score.className = 'scene-score';
      score.style.fontFamily = 'var(--font-mono)';
      score.style.color = 'var(--text-muted)';
      score.textContent = typeof c.similarity_score === 'number' ? c.similarity_score.toFixed(4) : '--';

      item.appendChild(label);
      item.appendChild(score);
      elements.imageSceneList.appendChild(item);
    });
  }

  // -------------------------------------------------------------------------
  // Dynamic Overview Metrics & Header Status Badges
  // -------------------------------------------------------------------------
  function updateOverviewMetrics() {
    const evidenceList = (AppState.currentCase && AppState.currentCase.evidence) ? AppState.currentCase.evidence : [];
    const evidenceCount = evidenceList.length;
    const eventCount = (AppState.fusion && AppState.fusion.events && AppState.fusion.events.length > 0)
      ? AppState.fusion.events.length
      : ((AppState.currentCase && AppState.currentCase.timeline_events) ? AppState.currentCase.timeline_events.length : 0);

    if (elements.statEvidenceCount) elements.statEvidenceCount.textContent = String(evidenceCount);
    if (elements.statEventCount) elements.statEventCount.textContent = String(eventCount);

    // Duration & dynamic labels
    if (AppState.videoResult && AppState.videoResult.duration_seconds !== null && AppState.videoResult.duration_seconds !== undefined) {
      if (elements.statDuration) elements.statDuration.textContent = `${AppState.videoResult.duration_seconds.toFixed(1)}s`;
      if (elements.statDurationLabel) elements.statDurationLabel.textContent = 'Video Duration';
      if (elements.statDurationSub) elements.statDurationSub.textContent = 'Active surveillance clip';
    } else if (AppState.evidence && AppState.evidence.duration !== null && AppState.evidence.duration !== undefined) {
      if (elements.statDuration) elements.statDuration.textContent = `${AppState.evidence.duration.toFixed(1)}s`;
      if (elements.statDurationLabel) elements.statDurationLabel.textContent = 'Audio Duration';
      if (elements.statDurationSub) elements.statDurationSub.textContent = 'Active acoustic file';
    } else {
      if (elements.statDuration) elements.statDuration.textContent = '—';
      if (elements.statDurationLabel) elements.statDurationLabel.textContent = 'Media Duration';
      if (elements.statDurationSub) elements.statDurationSub.textContent = 'No timed media active';
    }

    if (AppState.status === 'processing' || AppState.imageStatus === 'processing' || AppState.videoStatus === 'processing') {
      if (elements.statStatus) elements.statStatus.textContent = 'ANALYZING';
    } else {
      if (elements.statStatus) elements.statStatus.textContent = 'READY';
    }

    // Dynamic Header Modality Badges
    const audioEvs = evidenceList.filter((e) => e.media_type === 'audio');
    const imageEvs = evidenceList.filter((e) => e.media_type === 'image');
    const videoEvs = evidenceList.filter((e) => e.media_type === 'video');

    if (elements.headerAudioBadge) {
      if (audioEvs.length === 0) {
        elements.headerAudioBadge.textContent = 'Audio: None';
        elements.headerAudioBadge.className = 'badge badge-status-neutral';
      } else if (audioEvs.some((e) => e.status === 'completed')) {
        elements.headerAudioBadge.textContent = `Audio: ${audioEvs.length} Analyzed`;
        elements.headerAudioBadge.className = 'badge badge-status-ready';
      } else {
        elements.headerAudioBadge.textContent = `Audio: ${audioEvs.length} Pending`;
        elements.headerAudioBadge.className = 'badge badge-status-pending';
      }
    }

    if (elements.headerImageBadge) {
      if (imageEvs.length === 0) {
        elements.headerImageBadge.textContent = 'Images: None';
        elements.headerImageBadge.className = 'badge badge-status-neutral';
      } else if (imageEvs.some((e) => e.status === 'completed')) {
        elements.headerImageBadge.textContent = `Images: ${imageEvs.length} Analyzed`;
        elements.headerImageBadge.className = 'badge badge-status-ready';
      } else {
        elements.headerImageBadge.textContent = `Images: ${imageEvs.length} Pending`;
        elements.headerImageBadge.className = 'badge badge-status-pending';
      }
    }

    if (elements.headerVideoBadge) {
      if (videoEvs.length === 0) {
        elements.headerVideoBadge.textContent = 'Video: None';
        elements.headerVideoBadge.className = 'badge badge-status-neutral';
      } else if (videoEvs.some((e) => e.status === 'completed')) {
        elements.headerVideoBadge.textContent = `Video: ${videoEvs.length} Analyzed`;
        elements.headerVideoBadge.className = 'badge badge-status-ready';
      } else {
        elements.headerVideoBadge.textContent = `Video: ${videoEvs.length} Pending`;
        elements.headerVideoBadge.className = 'badge badge-status-pending';
      }
    }

    if (elements.footerCaseId) {
      elements.footerCaseId.textContent = AppState.currentCaseId || 'CASE-2026-MVP-001';
    }
  }

  function renderButtonState() {
    if (AppState.status === 'processing') {
      elements.btnAnalyze.disabled = true;
      elements.btnAnalyze.textContent = '⚡ Analyzing Audio...';
    } else {
      elements.btnAnalyze.textContent = '⚡ Analyze Audio Evidence';
      elements.btnAnalyze.disabled = !selectedFile;
    }
  }

  function renderImageButtonState() {
    if (AppState.imageStatus === 'processing') {
      elements.btnAnalyzeImage.disabled = true;
      elements.btnAnalyzeImage.textContent = '⚡ Analyzing Image...';
    } else {
      elements.btnAnalyzeImage.textContent = '⚡ Analyze Image Evidence';
      elements.btnAnalyzeImage.disabled = !selectedImageFile;
    }
  }

  function renderVideoButtonState() {
    if (AppState.videoStatus === 'processing') {
      elements.btnAnalyzeVideo.disabled = true;
      elements.btnAnalyzeVideo.textContent = '⚡ Analyzing Video...';
    } else {
      elements.btnAnalyzeVideo.textContent = '⚡ Analyze Video Evidence';
      elements.btnAnalyzeVideo.disabled = !selectedVideoFile;
    }
  }

  function showAlert(message, type) {
    clearAlert();
    const div = document.createElement('div');
    div.className = `alert alert-${type}`;
    div.textContent = message;
    elements.alertBox.appendChild(div);
  }

  function clearAlert() {
    while (elements.alertBox && elements.alertBox.firstChild) {
      elements.alertBox.removeChild(elements.alertBox.firstChild);
    }
  }

  // -------------------------------------------------------------------------
  // DOM Ready
  // -------------------------------------------------------------------------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
