// src/components/About.js
import { useEffect, useState } from 'react';
import { Check2, Clipboard, Envelope } from 'react-bootstrap-icons';
import apiClient from './appClient';
import './ApiDocs/ApiDocs.css';

const teamInstitutions = [
  {
    institution: 'Digital Metabolic Twin Center, University of Galway',
    location: 'Galway, Ireland',
    members: ['Saleh Alwer', 'Chidi Egwu', 'Farid Zare', 'Jack McGoldrick', 'Ronan Fleming'],
  },
  {
    institution: 'Faculty of Science, Technology and Medicine, University of Luxembourg',
    location: 'Belvaux, Luxembourg',
    members: ['Thomas Sauter', 'Hugues Escoffier'],
  },
  {
    institution: 'Systems Biology, Department of Life Sciences, Chalmers University of Technology',
    location: 'Gothenburg, Sweden',
    members: ['Eduard Kerkhoven'],
  },
  {
    institution: 'Luo Laboratory, Center for Synthetic Biochemistry, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences',
    location: 'Shenzhen, China',
    members: ['Han Yu', 'Xiaozhou Luo'],
  },
  {
    institution: 'Maranas Group, Department of Chemical Engineering, The Pennsylvania State University',
    location: 'University Park, PA, USA',
    members: ['Costas D. Maranas', 'Veda Boorla', 'Somtirtha Santra'],
  },
  {
    institution: 'Shanghai Zelixir Biotech Co. Ltd',
    location: 'Shanghai, China',
    members: ['Liangzhen Zheng'],
  },
  {
    institution: 'College of Computing and Data Science, Nanyang Technological University',
    location: 'Singapore',
    members: ['Zechen Wang'],
  },
  {
    institution: 'Töpfer Lab, Institute for Plant Sciences, University of Cologne',
    location: 'Cologne, Germany',
    members: ['Nadine Töpfer', 'Jan-Niklas Weder', 'Karim Taha'],
  },
];

const METRIC_CARDS = [
  { key: 'jobs_completed', label: 'Jobs' },
  { key: 'reactions_completed', label: 'Reactions' },
  { key: 'unique_protein_sequences', label: 'Distinct proteins' },
];

const PARAMETER_BREAKDOWN = [
  {
    key: 'kcat_predictions_completed',
    label: (
      <>
        <span className="about-math">k<sub>cat</sub></span>
      </>
    ),
  },
  {
    key: 'km_predictions_completed',
    label: (
      <>
        <span className="about-math">K<sub>M</sub></span>
      </>
    ),
  },
  {
    key: 'kcat_km_predictions_completed',
    label: (
      <>
        <span className="about-math-frac" aria-label="k sub cat over K sub M">
          <span className="about-math-frac__num">k<sub>cat</sub></span>
          <span className="about-math-frac__den">K<sub>M</sub></span>
        </span>
      </>
    ),
  },
];

const numberFormatter = new Intl.NumberFormat('en-US');
const ABOUT_STATS_STORAGE_KEY = 'about_stats_payload_v1';

const About = () => {
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState(() => {
    try {
      const raw = window.localStorage.getItem(ABOUT_STATS_STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  });
  const citationText = 'OpenKineticsPredictor: open-source platform for kinetic parameter prediction. Citation details to be added.';

  useEffect(() => {
    let isMounted = true;

    apiClient.get('/about-stats/')
      .then((response) => {
        if (!isMounted) return;
        const payload = response.data || {};
        setStats(payload);
        try {
          window.localStorage.setItem(ABOUT_STATS_STORAGE_KEY, JSON.stringify(payload));
        } catch {
          // Best effort only.
        }
      })
      .catch(() => {
        if (!isMounted) return;
        setStats(prev => prev ?? {});
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const getMetricValue = (key) => {
    const value = stats?.[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  };

  const formatMetric = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '--';
    return numberFormatter.format(value);
  };

  const parameterBreakdown = PARAMETER_BREAKDOWN.map((metric) => ({
    ...metric,
    value: getMetricValue(metric.key),
  }));

  const derivedParameterTotal = parameterBreakdown.every(metric => metric.value !== null)
    ? parameterBreakdown.reduce((sum, metric) => sum + metric.value, 0)
    : null;
  const parameterPredictionTotal =
    getMetricValue('parameter_predictions_completed') ?? derivedParameterTotal;

  const copyCitation = () => {
    if (!navigator.clipboard) return;

    navigator.clipboard.writeText(citationText)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy: ', err));
  };

  return (
    <div className="about-page">
      <div className="about-container container">
        <header className="about-header">
          <p className="about-eyebrow">Open source kinetic prediction consortium</p>
          <h1>OpenKineticsPredictor</h1>
          <p className="about-hero-copy">
            We developed this platform to make kinetic parameter prediction methods more accessible in an open source setting, so it can continue to expand as more methods are published and introduced.
          </p>
        </header>

        <section className="about-section about-metrics-section" aria-label="Platform usage metrics">
          <div className="about-section-heading">
            <p className="about-section-label">All-time usage</p>
            <h2>Platform activity</h2>
          </div>

          <div className="about-metrics-grid">
            {METRIC_CARDS.map((metric) => (
              <article key={metric.key} className="about-metric-card">
                <span className="about-metric-label">{metric.label}</span>
                <strong className="about-metric-value">{formatMetric(getMetricValue(metric.key))}</strong>
              </article>
            ))}

            <article className="about-metric-card about-metric-card--parameter">
              <div>
                <span className="about-metric-label">Parameter predictions</span>
                <strong className="about-metric-value">
                  {formatMetric(parameterPredictionTotal)}
                </strong>
              </div>
              <div className="about-parameter-breakdown" aria-label="Parameter prediction breakdown">
                {parameterBreakdown.map((metric) => (
                  <div key={metric.key} className="about-breakdown-row">
                    <span>{metric.label}</span>
                    <strong>{formatMetric(metric.value)}</strong>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        <section className="about-section about-consortium-section" aria-labelledby="about-consortium-title">
          <div className="about-section-heading">
            <p className="about-section-label">Consortium</p>
            <h2 id="about-consortium-title">Collaborating institutions</h2>
          </div>

          <div className="about-institution-grid">
            {teamInstitutions.map((entry, idx) => (
              <article key={entry.institution} className="about-institution-card">
                <div className="about-institution-heading">
                  <span className="about-institution-index">{String(idx + 1).padStart(2, '0')}</span>
                  <div>
                    <h3 className="about-institution-name">{entry.institution}</h3>
                    <p className="about-institution-location">{entry.location}</p>
                  </div>
                </div>
                <p className="about-member-list">{entry.members.join(', ')}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="about-section about-details-grid" aria-label="Contact and citation">
          <article className="about-detail-card">
            <div className="about-detail-heading">
              <h2>Contact</h2>
            </div>
            <p className="about-detail-text">
              For questions about the platform, collaborations, or contributing prediction methods.
            </p>
            <a href="mailto:s.alwer1@universityofgalway.ie" className="about-contact-link">
              <Envelope aria-hidden="true" />
              <span>s.alwer1@universityofgalway.ie</span>
            </a>
          </article>

          <article className="about-detail-card about-citation-card">
            <div className="about-detail-heading about-citation-heading">
              <h2>Citation</h2>
              <button type="button" className="about-copy-button" onClick={copyCitation}>
                {copied ? <Check2 aria-hidden="true" /> : <Clipboard aria-hidden="true" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
            <p className="about-citation-text">{citationText}</p>
          </article>
        </section>
      </div>
    </div>
  );
};

export default About;
