import React from 'react';
import './ApiDocs/ApiDocs.css';

const CONTRIBUTING_URL =
  'https://github.com/openKinetics/webKinPred/blob/master/docs/project/contributing.rst';

export default function Contribute() {
  return (
    <div className="api-docs-page">
      <div className="api-docs">
        <header className="api-docs-hero">
          <h1>Contribute a Method</h1>
          <p className="lead">
            This platform is open source, and we encourage contributors to add new kinetic prediction methods.
          </p>
        </header>

        <section className="api-docs-section">
          <p style={{ opacity: 0.9, lineHeight: 1.7 }}>
            Follow the contributor guide to add your method to the registry and make it available on this platform.
          </p>
          <p>
            <a
              href={CONTRIBUTING_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="contribute-link"
            >
              {CONTRIBUTING_URL}
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
