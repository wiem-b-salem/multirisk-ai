import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL, clearToken, getToken } from '../api'

const DISEASE_LABELS = { cvd: 'Cardiovascular', diabetes: 'Type 2 Diabetes' }

export default function History() {
  const [records, setRecords] = useState([])
  const navigate = useNavigate()

  useEffect(() => {
    async function load() {
      const response = await fetch(`${API_URL}/history`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })
      if (response.status === 401) {
        clearToken()
        navigate('/login')
        return
      }
      setRecords(await response.json())
    }
    load()
  }, [navigate])

  return (
    <div className="page">
      <h1>History</h1>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Disease</th>
            <th>Risk level</th>
            <th>Model score</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr key={record.id}>
              <td>{new Date(record.created_at).toLocaleString()}</td>
              <td>{DISEASE_LABELS[record.disease] || record.disease}</td>
              <td>{record.risk_level}</td>
              <td>{Math.round(record.risk_score * 100)} / 100</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
