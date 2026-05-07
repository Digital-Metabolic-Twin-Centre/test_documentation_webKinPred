// src/components/About.js
import { useEffect, useState } from 'react';
import { Button, Form } from 'react-bootstrap';
import apiClient from './appClient';
import './ApiDocs/ApiDocs.css';

const teamInstitutions = [
  {
    institution: 'Digital Metabolic Twin Center, University of Galway',
    location: 'Galway, Ireland',
    members: ['Ronan Fleming', 'Saleh Alwer'],
  },
  {
    institution: 'Faculty of Science, Technology and Medicine, University of Luxembourg',
    location: 'Esch-sur-Alzette, Luxembourg',
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
    members: ['Costas D. Maranas', 'Veda Sheersh Boorla', 'Somtirtha Santar'],
  },
  {
    institution: 'Shenzhen Zelixir Biotech Co. Ltd',
    location: 'Shenzhen, Guangdong, China',
    members: ['Liangzhen Zheng'],
  },
  {
    institution: 'School of Physics, Shandong University',
    location: 'Jinan, China',
    members: ['Zechen Wang'],
  },
  {
    institution: 'Töpfer Lab, Institute for Plant Sciences, University of Cologne',
    location: 'Cologne, Germany',
    members: ['Nadine Töpfer', 'Jan-Niklas Weder', 'Karim Taha'],
  },
];

const METRIC_CARDS = [
  { key: 'jobs_completed', label: 'Jobs processed' },
  { key: 'reactions_completed', label: 'Rows predicted' },
  { key: 'unique_protein_sequences', label: 'Distinct proteins' },
  {
    key: 'kcat_predictions_completed',
    label: (
      <>
        <span className="about-math">k<sub>cat</sub></span> predictions
      </>
    ),
  },
  {
    key: 'km_predictions_completed',
    label: (
      <>
        <span className="about-math">K<sub>M</sub></span> predictions
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
        </span>{' '}
        predictions
      </>
    ),
  },
];

const numberFormatter = new Intl.NumberFormat('en-US');

const About = () => {
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState(null);
  const citationText = "OpenKineticsPredictor: ....";

  useEffect(() => {
    let isMounted = true;

    apiClient.get('/about-stats/')
      .then((response) => {
        if (!isMounted) return;
        setStats(response.data || {});
      })
      .catch(() => {
        if (!isMounted) return;
        setStats({});
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const formatMetric = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return '—';
    return numberFormatter.format(value);
  };

  const copyCitation = () => {
    navigator.clipboard.writeText(citationText)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(err => console.error('Failed to copy: ', err));
  };

  return (
    <div className="api-docs-page">
      <div className="about-container container pt-4 pb-5">

        <div className="about-header">
          <h2>About OpenKineticsPredictor</h2>
        </div>

        <section className="about-metrics-section" aria-label="Platform usage metrics">
          <div className="about-metrics-grid">
            {METRIC_CARDS.map((metric) => (
              <article key={metric.key} className="about-metric-card">
                <div className="about-metric-value">{formatMetric(stats?.[metric.key])}</div>
                <div className="about-metric-label">{metric.label}</div>
              </article>
            ))}
          </div>
        </section>

        <section style={{ marginBottom: '3rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '1rem' }}>
            {teamInstitutions.map((entry, idx) => (
              <div key={idx} className="about-institution-card">
                <div>
                  <div className="about-institution-name">{entry.institution}</div>
                  <div className="about-institution-location">
                    <span style={{ opacity: 0.6 }}>&#x1F4CD;</span>
                    {entry.location}
                  </div>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.25rem' }}>
                  {entry.members.map((member, mIdx) => (
                    <span key={mIdx} className="about-member-badge">{member}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section style={{ marginBottom: '2rem' }}>
          <h4 className="about-citation-heading">Citation</h4>
          <Form.Group controlId="citationText">
            <Form.Control
              as="textarea"
              rows={3}
              readOnly
              value={citationText}
              className="about-citation-area"
            />
          </Form.Group>
          <Button variant="secondary" className="mt-2" onClick={copyCitation}>
            {copied ? 'Copied!' : 'Copy Citation'}
          </Button>
        </section>

      </div>
    </div>
  );
};

export default About;
