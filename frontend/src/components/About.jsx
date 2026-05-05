// src/components/About.js
import { useState } from 'react';
import { Button, Form } from 'react-bootstrap';
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

const About = () => {
  const [copied, setCopied] = useState(false);
  const citationText = "OpenKineticsPredictor: ....";

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
