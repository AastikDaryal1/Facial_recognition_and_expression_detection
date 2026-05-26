import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  TrendingUp, Users, Calendar, Clock, Smile, Award,
  Activity, ArrowLeft, RefreshCw, BarChart3, AlertCircle, ScanFace
} from 'lucide-react'
import { useAuth } from '../AuthContext'
import {
  fetchKPISummary,
  fetchKPISessionsOverTime,
  fetchKPIEmotionDistribution,
  fetchKPITopIdentified,
  fetchKPILatencyTrend,
  fetchKPIUserActivity,
  fetchKPIHourlyHeatmap
} from '../api'

export default function KPIDashboard() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // KPI Data States
  const [summary, setSummary] = useState(null)
  const [sessionsOverTime, setSessionsOverTime] = useState([])
  const [emotions, setEmotions] = useState([])
  const [topIdentified, setTopIdentified] = useState([])
  const [latencyTrend, setLatencyTrend] = useState([])
  const [userActivity, setUserActivity] = useState([])
  const [hourlyHeatmap, setHourlyHeatmap] = useState([])

  const loadData = async () => {
    setLoading(true)
    setError('')
    try {
      const [sum, sot, emo, top, lat, uact, heat] = await Promise.all([
        fetchKPISummary(days),
        fetchKPISessionsOverTime(days),
        fetchKPIEmotionDistribution(days),
        fetchKPITopIdentified(days, 8),
        fetchKPILatencyTrend(days),
        fetchKPIUserActivity(days, 5),
        fetchKPIHourlyHeatmap(days)
      ])
      setSummary(sum)
      setSessionsOverTime(sot)
      setEmotions(emo)
      setTopIdentified(top)
      setLatencyTrend(lat)
      setUserActivity(uact)
      setHourlyHeatmap(heat)
    } catch (err) {
      setError(err.message || 'Failed to load KPI analytics.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (user && (user.role === 'super_admin' || user.role === 'org_admin')) {
      loadData()
    } else if (user) {
      navigate('/')
    }
  }, [days, user])

  if (!user) return null

  // Helper to draw SVG Line Chart for Sessions/Latency Over Time
  const renderLineChart = (data, valueKey, labelKey, strokeColor, fillColor, isLatency = false) => {
    if (!data || data.length === 0) {
      return (
        <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)' }}>
          No data available for this range
        </div>
      )
    }

    const width = 500
    const height = 200
    const padding = 35

    const values = data.map(d => d[valueKey])
    const maxVal = Math.max(...values, 1)
    const minVal = Math.min(...values, 0)
    const valRange = maxVal - minVal

    const getX = (index) => padding + (index / (data.length - 1 || 1)) * (width - 2 * padding)
    const getY = (val) => height - padding - ((val - minVal) / valRange) * (height - 2 * padding)

    let points = ''
    let fillPoints = `${getX(0)},${height - padding} `

    data.forEach((d, idx) => {
      const x = getX(idx)
      const y = getY(d[valueKey])
      points += `${x},${y} `
      fillPoints += `${x},${y} `
    })

    fillPoints += `${getX(data.length - 1)},${height - padding}`

    return (
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%" style={{ overflow: 'visible' }}>
        <defs>
          <linearGradient id={`grad-${valueKey}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={fillColor} stopOpacity="0.4" />
            <stop offset="100%" stopColor={fillColor} stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
          const y = padding + ratio * (height - 2 * padding)
          const val = maxVal - ratio * valRange
          return (
            <g key={i}>
              <line
                x1={padding}
                y1={y}
                x2={width - padding}
                y2={y}
                stroke="rgba(148, 163, 184, 0.1)"
                strokeDasharray="4 4"
              />
              <text
                x={padding - 5}
                y={y + 4}
                fill="var(--muted)"
                fontSize="9"
                textAnchor="end"
              >
                {isLatency ? `${val.toFixed(2)}s` : Math.round(val)}
              </text>
            </g>
          )
        })}

        {/* Area fill */}
        <polygon points={fillPoints} fill={`url(#grad-${valueKey})`} />

        {/* Line */}
        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2.5"
          points={points}
        />

        {/* Interactive Dots */}
        {data.map((d, idx) => {
          const x = getX(idx)
          const y = getY(d[valueKey])
          // Only show labels/dots if reasonable density
          const showDetails = data.length <= 15 || idx % Math.ceil(data.length / 8) === 0
          if (!showDetails) return null

          return (
            <g key={idx} className="chart-dot-group">
              <circle
                cx={x}
                cy={y}
                r="4"
                fill="var(--bg)"
                stroke={strokeColor}
                strokeWidth="2"
              />
              <text
                x={x}
                y={height - 10}
                fill="var(--muted)"
                fontSize="9"
                textAnchor="middle"
              >
                {d[labelKey].substring(5)}
              </text>
            </g>
          )
        })}
      </svg>
    )
  }

  // Render SVG Donut Chart for Emotions
  const renderDonutChart = (data) => {
    if (!data || data.length === 0) {
      return (
        <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)' }}>
          No emotion data
        </div>
      )
    }

    const COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#f43f5e', '#64748b']
    const total = data.reduce((acc, curr) => acc + curr.count, 0)

    let accumulatedPercentage = 0

    const arcs = data.map((d, idx) => {
      const percentage = d.count / total
      const startAngle = accumulatedPercentage * 360
      const endAngle = (accumulatedPercentage + percentage) * 360
      accumulatedPercentage += percentage

      const color = COLORS[idx % COLORS.length]

      // Math for SVG Arc path
      const rad = Math.PI / 180
      const x1 = 100 + 70 * Math.cos((startAngle - 90) * rad)
      const y1 = 100 + 70 * Math.sin((startAngle - 90) * rad)
      const x2 = 100 + 70 * Math.cos((endAngle - 90) * rad)
      const y2 = 100 + 70 * Math.sin((endAngle - 90) * rad)

      const largeArc = percentage > 0.5 ? 1 : 0
      const pathData = percentage === 1
        ? `M 100 30 A 70 70 0 1 1 99.99 30`
        : `M ${x1} ${y1} A 70 70 0 ${largeArc} 1 ${x2} ${y2}`

      return {
        path: pathData,
        color,
        emotion: d.emotion,
        count: d.count,
        percent: Math.round(percentage * 100)
      }
    })

    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center' }}>
        <svg width="200" height="200" viewBox="0 0 200 200">
          {arcs.map((arc, idx) => (
            <path
              key={idx}
              d={arc.path}
              fill="none"
              stroke={arc.color}
              strokeWidth="20"
              style={{ transition: 'stroke-width 0.2s', cursor: 'pointer' }}
              title={`${arc.emotion}: ${arc.count} (${arc.percent}%)`}
            />
          ))}
          <circle cx="100" cy="100" r="60" fill="var(--bg-soft)" />
          <text x="100" y="95" fill="var(--text)" textAnchor="middle" fontWeight="bold" fontSize="18">
            {total}
          </text>
          <text x="100" y="115" fill="var(--muted)" textAnchor="middle" fontSize="10">
            TOTAL FACES
          </text>
        </svg>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: '150px' }}>
          {arcs.map((arc, idx) => (
            <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: arc.color }} />
              <span style={{ textTransform: 'capitalize', color: 'var(--text)' }}>{arc.emotion}</span>
              <span style={{ color: 'var(--muted)', marginLeft: 'auto' }}>
                {arc.count} ({arc.percent}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Render Heatmap (Hourly)
  const renderHourlyHeatmap = () => {
    if (!hourlyHeatmap || hourlyHeatmap.length === 0) return null

    const counts = hourlyHeatmap.map(h => h.count)
    const maxCount = Math.max(...counts, 1)

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--muted)' }}>
          <span>System Usage by Hour of Day</span>
          <span>Max: {maxCount} / hr</span>
        </div>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(24, 1fr)',
          gap: '4px',
          background: 'rgba(15,23,42,0.4)',
          padding: '10px',
          borderRadius: '8px',
          border: '1px solid var(--card-border)'
        }}>
          {hourlyHeatmap.map((h) => {
            const intensity = h.count / maxCount
            const opacity = 0.15 + intensity * 0.85
            const bg = h.count > 0 ? `rgba(99, 102, 241, ${opacity})` : 'rgba(148, 163, 184, 0.05)'
            const border = h.count > 0 ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid rgba(148, 163, 184, 0.1)'

            return (
              <div
                key={h.hour}
                title={`${h.hour}:00 - ${h.count} sessions`}
                style={{
                  aspectRatio: '1',
                  background: bg,
                  border: border,
                  borderRadius: '3px',
                  position: 'relative',
                  cursor: 'pointer'
                }}
              />
            )
          })}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--muted)' }}>
          <span>12 AM</span>
          <span>6 AM</span>
          <span>12 PM</span>
          <span>6 PM</span>
          <span>11 PM</span>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', paddingBottom: '3rem' }}>
      {/* Header Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button
            onClick={() => navigate(user.role === 'super_admin' ? '/admin' : '/org-dashboard')}
            className="secondary-btn"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem' }}
          >
            <ArrowLeft size={16} />
            Back
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 className="text-accent" />
              Analytics & KPIs
            </h1>
            <p style={{ margin: 0, color: 'var(--muted)', fontSize: '0.9rem' }}>
              System utilization, latency profiles, and face identification rate analytics.
            </p>
          </div>
        </div>

        {/* Date Filter & Refresh */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Calendar size={16} style={{ position: 'absolute', left: '10px', color: 'var(--muted)' }} />
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="glass-card"
              style={{
                padding: '0.5rem 1rem 0.5rem 2rem',
                border: '1px solid var(--card-border)',
                borderRadius: '8px',
                color: 'var(--text)',
                outline: 'none',
                cursor: 'pointer',
                appearance: 'none',
                paddingRight: '2rem'
              }}
            >
              <option value="7">Last 7 Days</option>
              <option value="30">Last 30 Days</option>
              <option value="90">Last 90 Days</option>
              <option value="365">Last Year</option>
            </select>
          </div>

          <button
            onClick={loadData}
            disabled={loading}
            className="secondary-btn"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem' }}
          >
            <RefreshCw size={16} className={loading ? 'spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="glass-card" style={{ padding: '1rem', border: '1px solid var(--danger)', display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'rgba(251, 113, 133, 0.1)' }}>
          <AlertCircle color="var(--danger)" />
          <span style={{ color: 'var(--danger)' }}>{error}</span>
        </div>
      )}

      {/* Summary Stat Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem'
      }}>
        <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', borderRadius: '10px', padding: '0.75rem' }}>
            <Activity size={24} />
          </div>
          <div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Total Sessions</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{summary?.total_sessions ?? '—'}</div>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ background: 'rgba(34, 211, 238, 0.15)', color: '#22d3ee', borderRadius: '10px', padding: '0.75rem' }}>
            <ScanFace size={24} />
          </div>
          <div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Faces Processed</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{summary?.total_faces ?? '—'}</div>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#4ade80', borderRadius: '10px', padding: '0.75rem' }}>
            <Award size={24} />
          </div>
          <div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Identified Faces</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{summary?.total_identified ?? '—'}</div>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', borderRadius: '10px', padding: '0.75rem' }}>
            <TrendingUp size={24} />
          </div>
          <div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Identification Rate</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{summary?.identification_rate ? `${summary.identification_rate}%` : '—'}</div>
          </div>
        </div>

        <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ background: 'rgba(236, 72, 153, 0.15)', color: '#f472b6', borderRadius: '10px', padding: '0.75rem' }}>
            <Clock size={24} />
          </div>
          <div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Avg Latency</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800 }}>{summary?.avg_latency_s ? `${summary.avg_latency_s}s` : '—'}</div>
          </div>
        </div>
      </div>

      {/* Main KPI Charts Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>
        {/* Sessions over time */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Inferences Over Time</h3>
          <div style={{ height: '220px' }}>
            {renderLineChart(sessionsOverTime, 'count', 'date', '#6366f1', '#6366f1')}
          </div>
        </div>

        {/* Latency profile over time */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Avg Processing Latency</h3>
          <div style={{ height: '220px' }}>
            {renderLineChart(latencyTrend, 'avg_latency_s', 'date', '#ec4899', '#ec4899', true)}
          </div>
        </div>

        {/* Emotion Breakdown */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Detected Emotion Distribution</h3>
          <div style={{ minHeight: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {renderDonutChart(emotions)}
          </div>
        </div>

        {/* Top identified / Leaderboard / Heatmap panel */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Usage Profile & Heatmap</h3>
          {renderHourlyHeatmap()}
        </div>
      </div>

      {/* Bottom Leaderboards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '1.5rem' }}>
        {/* Top identified individuals */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Most Frequently Identified Persons</h3>
          {topIdentified.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--muted)' }}>No persons identified yet</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {topIdentified.map((p, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--muted)', width: '20px' }}>#{idx + 1}</span>
                  <span style={{ fontWeight: 600, flex: 1 }}>{p.name}</span>
                  <div style={{ width: '120px', background: 'rgba(148,163,184,0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${(p.count / topIdentified[0].count) * 100}%`,
                      background: 'var(--accent-blue)',
                      height: '100%'
                    }} />
                  </div>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, minWidth: '40px', textAlign: 'right' }}>
                    {p.count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Active user leaderboard */}
        <div className="glass-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700 }}>Most Active Members</h3>
          {userActivity.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--muted)' }}>No user activity recorded</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--card-border)', color: 'var(--muted)', textAlign: 'left' }}>
                  <th style={{ padding: '0.5rem 0' }}>User</th>
                  <th style={{ padding: '0.5rem 0', textAlign: 'center' }}>Sessions</th>
                  <th style={{ padding: '0.5rem 0', textAlign: 'right' }}>Faces Processed</th>
                </tr>
              </thead>
              <tbody>
                {userActivity.map((act) => (
                  <tr key={act.user_id} style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.05)' }}>
                    <td style={{ padding: '0.75rem 0', fontWeight: 500, color: 'var(--text)' }}>
                      {act.email}
                    </td>
                    <td style={{ padding: '0.75rem 0', textAlign: 'center', fontWeight: 700, color: 'var(--accent-indigo)' }}>
                      {act.session_count}
                    </td>
                    <td style={{ padding: '0.75rem 0', textAlign: 'right', color: 'var(--muted)' }}>
                      {act.total_faces}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
