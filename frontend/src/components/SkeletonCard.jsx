/**
 * src/components/SkeletonCard.jsx
 * Loading skeleton for face result cards.
 */

export default function SkeletonCard() {
  return (
    <div className="skeleton-card fade-in-up">
      <div className="skeleton-avatar"></div>
      <div className="skeleton-text-block">
        <div className="skeleton-text short"></div>
        <div className="skeleton-text long"></div>
      </div>
      <div className="skeleton-text short" style={{ width: '60px' }}></div>
    </div>
  )
}
