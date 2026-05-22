import React, { useState, useMemo, useRef, useCallback } from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Accordion, Row, Col } from 'react-bootstrap';
import { InfoCircle, Diagram3, GearFill, Download } from 'react-bootstrap-icons';
import { useTheme } from '../context/ThemeContext';
import '../styles/components/SequenceSimilarityHistogram.css';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const GRANULARITY_BINS = [1, 2, 4, 5, 10, 20, 25, 50, 100];

function SequenceSimilarityHistogram({ similarityData }) {
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const axisColor = isDark ? 'rgba(255,255,255,0.75)' : '#2d2060';
  const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(110,90,200,0.15)';

  const chartRef = useRef(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const models = similarityData ? Object.keys(similarityData) : [];
  const [activeModel, setActiveModel] = useState(models[0] || '');
  const [similarityType, setSimilarityType] = useState('max');

  const DEFAULT_BIN_INDEX = 3;
  const [binIndex, setBinIndex] = useState(DEFAULT_BIN_INDEX);
  const numberOfBins = GRANULARITY_BINS[binIndex];

  const { labels, dataValues, countValues } = useMemo(() => {
    if (!similarityData || !activeModel || !similarityData[activeModel]) {
      return { labels: [], dataValues: [], countValues: [] };
    }

    const modelData = similarityData[activeModel];
    const rawPercentages = similarityType === 'mean' ? modelData.histogram_mean : modelData.histogram_max;
    const rawCounts = similarityType === 'mean' ? modelData.count_mean : modelData.count_max;

    if (!rawPercentages || !rawCounts) {
      return { labels: [], dataValues: [], countValues: [] };
    }

    if (numberOfBins >= 100) {
      const sortedLabels = Object.keys(rawPercentages).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
      return {
        labels: sortedLabels,
        dataValues: sortedLabels.map(label => rawPercentages[label]),
        countValues: sortedLabels.map(label => rawCounts[label]),
      };
    }

    const newLabels = [];
    const newDataValues = [];
    const newCountValues = [];
    const binSize = 101 / numberOfBins;

    for (let i = 0; i < numberOfBins; i++) {
      const start = Math.floor(i * binSize);
      const end = Math.floor((i + 1) * binSize) - 1;
      const actualEnd = i === numberOfBins - 1 ? 100 : end;

      let binPercentage = 0;
      let binCount = 0;
      for (let j = start; j <= actualEnd; j++) {
        const key = String(j);
        if (rawPercentages[key] !== undefined) binPercentage += rawPercentages[key];
        if (rawCounts[key] !== undefined) binCount += rawCounts[key];
      }
      newLabels.push(`${start}-${actualEnd}`);
      newDataValues.push(binPercentage);
      newCountValues.push(binCount);
    }

    return { labels: newLabels, dataValues: newDataValues, countValues: newCountValues };
  }, [activeModel, similarityData, similarityType, numberOfBins]);

  // Derived early (before the null guard) so they're in scope for the useCallback deps
  const modelData = (similarityData && similarityData[activeModel]) || {};
  const averageSimilarity = similarityType === 'mean'
    ? modelData.average_mean_similarity
    : modelData.average_max_similarity;

  const handleDownload = useCallback(async () => {
    if (isDownloading) return;
    setIsDownloading(true);

    try {
      await new Promise(resolve => requestAnimationFrame(resolve));

      // ── Render a fresh off-screen chart at fixed high resolution ──
      // This avoids all blurriness that comes from screenshotting the live canvas.
      const S          = 2;          // internal pixel scale
      const CHART_W    = 1200;       // logical CSS width
      const CHART_H    = 500;        // logical CSS height

      const typeLabel     = similarityType === 'max' ? 'Max' : 'Mean';
      const exportAxis    = isDark ? 'rgba(255,255,255,0.80)' : '#2d2060';
      const exportGrid    = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(110,90,200,0.15)';

      const offCanvas = document.createElement('canvas');
      offCanvas.width  = CHART_W;   // Chart.js will multiply by devicePixelRatio
      offCanvas.height = CHART_H;

      const exportChart = new ChartJS(offCanvas, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: dataValues,
            backgroundColor: 'rgba(75,192,192,0.45)',
            borderColor:     'rgba(75,192,192,1)',
            borderWidth: 1,
            barPercentage: 1.0,
            categoryPercentage: 0.9,
          }],
        },
        options: {
          responsive: false,
          animation: false,
          devicePixelRatio: S,
          plugins: {
            legend:  { display: false },
            tooltip: { enabled: false },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: `${typeLabel} Sequence Similarity (%)`,
                color: exportAxis,
                font: { size: 13, family: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', weight: '500' },
              },
              ticks: { color: exportAxis, font: { size: 11 } },
              grid:  { color: exportGrid },
            },
            y: {
              title: {
                display: true,
                text: 'Frequency of Input Sequences (%)',
                color: exportAxis,
                font: { size: 13, family: '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', weight: '500' },
              },
              ticks: { color: exportAxis, font: { size: 11 } },
              grid:  { color: exportGrid },
              beginAtZero: true,
            },
          },
        },
      });

      // offCanvas is now S× physical pixels (2400 × 1000)
      const src = offCanvas;

      // ── Composite canvas layout (all in physical pixels = logical × S) ──
      const hPad    = 36 * S;
      const titleH  = 80 * S;
      const footerH = 52 * S;

      const dest = document.createElement('canvas');
      dest.width  = src.width  + hPad * 2;
      dest.height = src.height + titleH + footerH;

      const ctx = dest.getContext('2d');

      // ── Background ──────────────────────────────────────────────
      ctx.fillStyle = isDark ? '#1a1633' : '#f5f2ff';
      ctx.fillRect(0, 0, dest.width, dest.height);

      const grad = ctx.createLinearGradient(0, 0, 0, dest.height);
      grad.addColorStop(0, isDark ? 'rgba(110,90,200,0.10)' : 'rgba(110,90,200,0.07)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, dest.width, dest.height);

      // ── Title block ─────────────────────────────────────────────
      const titleColor = isDark ? 'rgba(255,255,255,0.92)' : '#1c1443';
      const mutedColor = isDark ? 'rgba(255,255,255,0.45)' : 'rgba(44,32,100,0.50)';

      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';

      ctx.font      = `600 ${15 * S}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
      ctx.fillStyle = titleColor;
      ctx.fillText(
        `Input sequences vs. ${activeModel} training data`,
        dest.width / 2, titleH * 0.36,
      );

      ctx.font      = `400 ${11 * S}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
      ctx.fillStyle = mutedColor;
      ctx.fillText(
        `${typeLabel} Sequence Identity Distribution`,
        dest.width / 2, titleH * 0.72,
      );

      // ── Chart ────────────────────────────────────────────────────
      ctx.drawImage(src, hPad, titleH);

      // ── Bar value labels ─────────────────────────────────────────
      // bar.x / bar.y are in Chart.js logical-pixel space (0..CHART_W / 0..CHART_H);
      // multiply by S to convert to physical pixel space on the destination canvas.
      if (dataValues.length <= 25) {
        const meta     = exportChart.getDatasetMeta(0);
        const barColor = isDark ? 'rgba(100,210,210,1)' : 'rgba(14,90,90,1)';

        ctx.font         = `600 ${9.5 * S}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'bottom';
        ctx.shadowBlur   = 4 * S;
        ctx.shadowColor  = isDark ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.95)';
        ctx.fillStyle    = barColor;

        meta.data.forEach((bar, i) => {
          const value = dataValues[i];
          if (!value || value < 0.5) return;
          ctx.fillText(
            `${value.toFixed(1)}%`,
            hPad + bar.x * S,
            titleH + bar.y * S - 5 * S,
          );
        });

        ctx.shadowBlur  = 0;
        ctx.shadowColor = 'transparent';
      }

      // ── Footer ───────────────────────────────────────────────────
      const footerTopY = src.height + titleH;
      ctx.strokeStyle  = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(110,90,200,0.18)';
      ctx.lineWidth    = 1 * S;
      ctx.beginPath();
      ctx.moveTo(hPad, footerTopY + 10 * S);
      ctx.lineTo(dest.width - hPad, footerTopY + 10 * S);
      ctx.stroke();

      const midFooterY = footerTopY + footerH * 0.60;

      if (averageSimilarity !== null && averageSimilarity !== undefined) {
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'middle';
        ctx.font         = `500 ${10.5 * S}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
        ctx.fillStyle    = isDark ? 'rgba(255,255,255,0.50)' : 'rgba(44,32,100,0.52)';
        ctx.fillText(
          `Average ${typeLabel.toLowerCase()} similarity: ${Number(averageSimilarity).toFixed(1)}%`,
          hPad, midFooterY,
        );
      }

      ctx.textAlign    = 'right';
      ctx.textBaseline = 'middle';
      ctx.font         = `400 ${10 * S}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
      ctx.fillStyle    = isDark ? 'rgba(255,255,255,0.22)' : 'rgba(44,32,100,0.28)';
      ctx.fillText('predictor.openkinetics.org', dest.width - hPad, midFooterY);

      // ── Clean up + trigger download ───────────────────────────────
      exportChart.destroy();

      const url  = dest.toDataURL('image/png');
      const link = document.createElement('a');
      link.href  = url;
      link.download = `similarity_${activeModel.replace(/[\s/\\:*?"<>|]+/g, '-')}_${similarityType}.png`;
      link.click();
    } finally {
      setIsDownloading(false);
    }
  }, [isDownloading, activeModel, similarityType, isDark, labels, dataValues, averageSimilarity]);

  if (!similarityData) return null;

  const data = {
    labels,
    datasets: [{
      label: `% of input at similarity`,
      data: dataValues,
      backgroundColor: 'rgba(75,192,192,0.4)',
      borderColor: 'rgba(75,192,192,1)',
      borderWidth: 1,
      barPercentage: 1.0,
      categoryPercentage: 0.9,
    }],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        callbacks: {
          title: (tooltipItems) => `Similarity Range: ${tooltipItems[0].label}%`,
          label: (context) => {
            const percentage = context.parsed.y;
            const count = countValues[context.dataIndex];
            return `${percentage.toFixed(1)}% of input (${count} sequence${count !== 1 ? 's' : ''}) found in this range`;
          },
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: `${similarityType === 'mean' ? 'Mean' : 'Max'} Sequence Similarity (%)`, color: axisColor, font: { size: 14 } },
        ticks: { color: axisColor, font: { size: 12 } },
        grid: { color: gridColor },
      },
      y: {
        title: { display: true, text: 'Frequency of Your Input Sequences (%)', color: axisColor, font: { size: 14 } },
        ticks: { color: axisColor, font: { size: 12 } },
        grid: { color: gridColor },
        beginAtZero: true,
      },
    },
  };

  return (
    <div>
      <div className="mt-4">
        <h5 className="sim-section-label text-center mb-3">Chart Controls</h5>
        <div className="p-3 rounded sim-controls-wrapper">
          <div className="row justify-content-between align-items-center mb-4">
            <div className="col-md-auto mb-3 mb-md-0">
              <label className="form-label d-block mb-2 small sim-section-label">Dataset</label>
              <div>
                {models.map(model => (
                  <button
                    key={model}
                    onClick={() => setActiveModel(model)}
                    className={`btn btn-kave-toggle ${activeModel === model ? 'active' : ''} me-2`}
                  >
                    {model}
                  </button>
                ))}
              </div>
            </div>
            <div className="col-md-auto">
              <label className="form-label d-block mb-2 small sim-section-label">View Type</label>
              <div className="btn-group" role="group">
                <button
                  type="button"
                  className={`btn sim-view-btn ${similarityType === 'max' ? 'active' : ''}`}
                  onClick={() => setSimilarityType('max')}
                >
                  Max
                </button>
                <button
                  type="button"
                  className={`btn sim-view-btn ${similarityType === 'mean' ? 'active' : ''}`}
                  onClick={() => setSimilarityType('mean')}
                >
                  Mean
                </button>
              </div>
            </div>
          </div>

          <div className="row justify-content-center">
            <div className="col-lg-7 d-flex align-items-center sim-slider-row">
              <label htmlFor="granularity-slider" className="form-label me-3 mb-0 small sim-section-label">
                Granularity
              </label>
              <input
                type="range"
                className="form-range"
                id="granularity-slider"
                min="0"
                max={GRANULARITY_BINS.length - 1}
                value={binIndex}
                onChange={(e) => setBinIndex(parseInt(e.target.value, 10))}
              />
              <span className="fw-bold ms-3" style={{ minWidth: '70px', textAlign: 'right' }}>
                {numberOfBins} Bins
              </span>
            </div>
          </div>
        </div>
      </div>

      {averageSimilarity !== null && (
        <div className="d-flex align-items-center justify-content-center gap-2 my-4">
          <p className="sim-body-text mb-0">
            Input sequences vs. <strong>{activeModel}</strong> training data ({similarityType} similarity)
          </p>
          <button
            className="btn sim-download-btn"
            onClick={handleDownload}
            disabled={isDownloading}
            title="Download chart as PNG"
            aria-label="Download chart as PNG"
          >
            {isDownloading
              ? <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
              : <Download size={15} />
            }
          </button>
        </div>
      )}

      <Bar ref={chartRef} data={data} options={options} />

      <h5 className="tab-section-header text-center mt-5 mb-4">Sequence Similarity Histogram</h5>

      <Accordion defaultActiveKey="0" className="custom-accordion">
        <Accordion.Item eventKey="0">
          <Accordion.Header>
            <InfoCircle className="me-2" /> What does this chart show?
          </Accordion.Header>
          <Accordion.Body>
            This histogram displays the distribution of sequence similarities found when searching your input protein sequences against the training data of a selected model. This helps you understand how novel or similar your sequences are compared to the data the model was trained on, which can influence prediction confidence.
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="1">
          <Accordion.Header>
            <Diagram3 className="me-2" /> Understanding Similarity Types
          </Accordion.Header>
          <Accordion.Body>
            You can view the similarity distribution in two ways:
            <Row className="mt-2">
              <Col md={6}>
                <strong>Max Similarity</strong>
                <p className="sim-section-label small">The single highest percentage identity found for each of your sequences. This represents the &quot;best match&quot; in the training data.</p>
              </Col>
              <Col md={6}>
                <strong>Mean Similarity</strong>
                <p className="sim-section-label small">The average percentage identity calculated from all significant alignment hits found for each of your sequences.</p>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="2">
          <Accordion.Header>
            <GearFill className="me-2" /> Technical Details &amp; Parameters
          </Accordion.Header>
          <Accordion.Body>
            <p>If no significant hits are found for a sequence, both its mean and max similarity are set to 0%. All similarity values are rounded to the nearest integer for binning.</p>
            <strong>MMseqs2 Parameters Used:</strong>
            <ul className="list-unstyled ps-3 mt-1 parameter-list">
              <li><code>-s</code> (sensitivity): <code>7.5</code></li>
              <li><code>-e</code> (E-value): <code>0.001</code></li>
              <li><code>--max-seqs</code>: <code>5000</code></li>
            </ul>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    </div>
  );
}

export default SequenceSimilarityHistogram;
