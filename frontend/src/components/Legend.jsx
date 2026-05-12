/**
 * src/components/Legend.jsx
 * Emotion color legend + identity style legend.
 * Extracted unchanged from original App.jsx.
 */

import { HelpCircle, User } from 'lucide-react'

const emotionColors = {
  Angry: '#FF4C4C', Fear: '#8E44AD', Happy: '#FFD93D',
  Neutral: '#BDC3C7', Sad: '#3498DB', Surprise: '#FF9F43',
}

export default function Legend() {
  return (
    <div className="dual-legend glass-card fade-in-up">
      <div className="legend-section">
        <h4 className="legend-title">Emotion (Color)</h4>
        <div className="legend-items">
          {Object.entries(emotionColors).map(([emotion, color]) => (
            <div key={emotion} className="legend-item">
              <span className="legend-color" style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}></span>
              <span className="legend-text">{emotion}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="legend-divider"></div>
      <div className="legend-section">
        <h4 className="legend-title">Identity (Style)</h4>
        <div className="legend-items">
          <div className="legend-item style-known">
            <span className="legend-style-box solid"></span>
            <span className="legend-text"><User size={14} className="inline-icon" /> Known</span>
          </div>
          <div className="legend-item style-unknown">
            <span className="legend-style-box dashed"></span>
            <span className="legend-text"><HelpCircle size={14} className="inline-icon" /> Unknown</span>
          </div>
        </div>
      </div>
    </div>
  )
}
