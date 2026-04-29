// src/components/HowToUseCard.js

import { useState } from 'react';
import { Card, Row, Col, Alert, Button } from 'react-bootstrap';
import { BoxArrowInDown, Bullseye, CloudUpload, Cpu, Github, ChevronDown, BoxArrowUpRight } from 'react-bootstrap-icons';
import '../../../styles/components/HowToUseCard.css';


export default function HowToUseCard({ methods = {} }) {
  const [openKeys, setOpenKeys] = useState(new Set());

  const toggle = (key) => setOpenKeys(prev => {
    const next = new Set(prev);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  });

  const methodEntries = Object.entries(methods).sort(([, a], [, b]) =>
    (a.displayName || '').localeCompare(b.displayName || '')
  );
  const cols = [[], [], []];
  methodEntries.forEach((entry, i) => cols[i % 3].push(entry));

  const targetLabel = {
    kcat: 'kcat',
    Km: 'Km',
    'kcat/Km': 'kcat/Km',
  };

  return (
    <Card className="section-container how-to-use-card mb-4">
      <Card.Header as="h3" className="text-center">
        How to Use This Tool
      </Card.Header>
      <Card.Body>
        <p className="lead text-center mb-4">
          Predict kinetic parameters (k<sub>cat</sub>, K<sub>M</sub>, and k<sub>cat</sub>/K<sub>M</sub>) for enzyme-catalysed reactions using various machine learning models.
        </p>
        <Alert variant="info" className="d-flex align-items-center">
          <Bullseye size={24} className="me-3" />
          <div>
            Ticking <strong>“Prefer experimental data”</strong> will first search BRENDA, SABIO-RK, and UniProt for known values. If found, these are used instead of model predictions.
          </div>
        </Alert>

        <Row className="text-center">
          <Col md={4} className="step-col">
            <div className="step-icon"><Bullseye size={30} /></div>
            <h5>Step 1: Select Prediction</h5>
            <p>Choose one or more targets: k<sub>cat</sub>, K<sub>M</sub>, and/or k<sub>cat</sub>/K<sub>M</sub>.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><CloudUpload size={30} /></div>
            <h5>Step 2: Upload Data</h5>
            <p>Provide your reaction data by uploading a formatted CSV file.</p>
          </Col>
          <Col md={4} className="step-col">
            <div className="step-icon"><Cpu size={30} /></div>
            <h5>Step 3: Choose Method</h5>
            <p>Select your desired prediction model(s) after optional validation.</p>
          </Col>
        </Row>

        <hr className="my-4" />

        <h4 className="text-center mb-3">Available Predictors</h4>
        <div className="mpill-grid">
          {cols.map((col, ci) => (
            <div key={ci} className="mpill-col">
              {col.map(([key, details]) => {
                const isOpen = openKeys.has(key);
                return (
                  <div key={key} className={`mpill${isOpen ? ' mpill--open' : ''}`}>
                    <button
                      type="button"
                      className="mpill-header"
                      onClick={() => toggle(key)}
                      aria-expanded={isOpen}
                    >
                      <span className="mpill-name">{details.displayName}</span>
                      <div className="mpill-right">
                        <div className="mpill-chips">
                          {(details.supports || []).map((target) => (
                            <span key={target} className="mpill-chip">
                              {targetLabel[target] || target}
                            </span>
                          ))}
                        </div>
                        <ChevronDown size={13} className="mpill-chevron" />
                      </div>
                    </button>

                    <div className="mpill-body">
                      <div className="mpill-body-inner">
                        <div className="mpill-pub">
                          {details.citationUrl ? (
                            <a
                              href={details.citationUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="mpill-pub-link"
                            >
                              {details.publicationTitle}
                              <BoxArrowUpRight size={10} className="mpill-ext-icon" />
                            </a>
                          ) : (
                            <span className="mpill-pub-title">{details.publicationTitle}</span>
                          )}
                        </div>

                        {details.authors && (
                          <p className="mpill-authors">{details.authors}</p>
                        )}

                        {details.moreInfo && (
                          <p className="mpill-note">
                            <span className="mpill-note-kw">Note</span>
                            {details.moreInfo}
                          </p>
                        )}

                        {details.repoUrl && (
                          <a
                            href={details.repoUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mpill-github"
                          >
                            <Github size={13} />
                            View on GitHub
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <hr className="my-4" />
        <h4 className="text-center mb-3">Input Data Format</h4>
        <p className="fmtt-subtitle">Required CSV columns for each format</p>
        <div className="fmttable">
          {/* Column headers */}
          <div className="fmtt-row fmtt-head">
            <div className="fmtt-label-cell" />
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Single-Substrate</span>
              <span className="fmtt-models">DLKcat · EITLEM · UniKP · KinForm-H · KinForm-L · CataPro · CatPred (K<sub>M</sub>)</span>
            </div>
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Multi-Substrate</span>
              <span className="fmtt-models">CatPred (k<sub>cat</sub> only)</span>
            </div>
            <div className="fmtt-cell fmtt-col-head">
              <span className="fmtt-format-name">Full Reaction</span>
              <span className="fmtt-models">TurNup</span>
            </div>
          </div>

          {/* Protein Sequence */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Protein Sequence</code>
            </div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
            <div className="fmtt-cell fmtt-present">Full amino-acid sequence</div>
          </div>

          {/* Substrate (singular) */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Substrate</code>
            </div>
            <div className="fmtt-cell fmtt-present"><code>SMILES</code> or <code>InChI</code> — one per row</div>
            <div className="fmtt-cell fmtt-present">Co-substrates joined with <code>.</code> <span className="fmtt-example">e.g. CC(=O)O.O</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
          </div>

          {/* Substrates (plural) */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Substrates</code>
            </div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-present">Semicolon-separated <code>SMILES</code> or <code>InChI</code></div>
          </div>

          {/* Products */}
          <div className="fmtt-row">
            <div className="fmtt-label-cell">
              <code className="fmtt-label">Products</code>
            </div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-absent"><span className="fmtt-not-required">not applicable</span></div>
            <div className="fmtt-cell fmtt-present">Semicolon-separated <code>SMILES</code> or <code>InChI</code></div>
          </div>
        </div>

        <hr className="my-4" />
        <h4 className="text-center mb-3">Example Templates</h4>
        <div className="d-grid gap-2 d-md-flex justify-content-md-center">
          <Button
            href="/templates/single_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Single-Substrate Template
          </Button>

          <Button
            href="/templates/multi_substrate_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Multi-Substrate Template
          </Button>

          <Button
            href="/templates/full_reaction_template.csv"
            download
            className="btn btn-custom-subtle"
          >
            <BoxArrowInDown className="me-2" />
            Full-Reaction Template
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
}
