/**
 * src/components/FaceCard.jsx
 * Individual face result card.
 * Extracted unchanged from original App.jsx.
 */

import { HelpCircle, User } from 'lucide-react'

const emotionColors = {
  Angry: '#FF4C4C', Fear: '#8E44AD', Happy: '#FFD93D',
  Neutral: '#BDC3C7', Sad: '#3498DB', Surprise: '#FF9F43',
}

function getEmotionColor(emotion) {
  if (!emotion) return '#BDC3C7'
  const match = Object.keys(emotionColors).find(
    (e) => e.toLowerCase() === emotion.toLowerCase()
  )
  return match ? emotionColors[match] : '#BDC3C7'
}

function FaceAvatar({ face, color }) {
  const isUnknown = !face.name || face.name.toUpperCase() === 'UNKNOWN'
  const getImage = (person) => person.face_image || '/dataset/unknown.png'
  return (
    <div
      className={`avatar-square ${isUnknown ? 'unknown-avatar' : 'known-avatar'}`}
      style={{ borderColor: color, boxShadow: `0 0 10px ${color}80`, borderStyle: isUnknown ? 'dashed' : 'solid' }}
    >
      <img
        src={getImage(face)}
        alt={face.name || 'Unknown'}
        onError={(e) => { e.target.src = '/dataset/unknown.png' }}
        className="avatar-image"
      />
    </div>
  )
}

export default function FaceCard({ face, delay, onHover, isHighlighted }) {
  const isUnknown = !face.name || face.name.toLowerCase() === 'unknown'
  const color = getEmotionColor(face.emotion)

  return (
    <div
      className={`face-card fade-in-up ${isUnknown ? 'face-card-unknown' : 'face-card-known'} ${isHighlighted ? 'highlighted' : ''}`}
      onMouseEnter={() => onHover && onHover(face.face_idx !== undefined ? face.face_idx : face.id)}
      onMouseLeave={() => onHover && onHover(null)}
      style={{
        borderColor: isHighlighted ? '#fff' : color,
        borderStyle: isUnknown ? 'dashed' : 'solid',
        boxShadow: isHighlighted ? `0 0 20px ${color}` : `0 0 8px ${color}33`,
        animationDelay: delay ? `${delay}s` : '0s',
        transform: isHighlighted ? 'translateY(-5px) scale(1.02)' : 'none',
        filter: isHighlighted ? 'brightness(1.2)' : 'none',
        zIndex: isHighlighted ? 10 : 1,
      }}
    >
      <div className="face-card-left">
        <FaceAvatar face={face} color={color} />
      </div>
      <div className="face-card-middle">
        <p style={{ color: isUnknown ? '#ffffff' : 'inherit' }}>
          <strong>Name:</strong> {isUnknown ? 'UNKNOWN' : face.name}
        </p>
        <p>
          <strong>Emotion:</strong> <span style={{ color }}>{face.emotion}</span>
        </p>
      </div>
      <div className="face-card-right">
        {isUnknown ? (
          <span className="identity-badge unknown"><HelpCircle size={14} /> Unknown</span>
        ) : (
          <span className="identity-badge known"><User size={14} /> Known</span>
        )}
      </div>
    </div>
  )
}
