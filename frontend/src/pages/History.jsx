import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { API_URL, clearToken, getToken } from '../api'

const DISEASE_LABELS = {
  cvd: 'Cardiovascular',
  diabetes: 'Type 2 Diabetes',
  alzheimers: "Alzheimer's disease",
  liver: 'Liver disease',
}

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
            <th>ML result</th>
            <th>DL result</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr key={record.id}>
              <td>{new Date(record.created_at).toLocaleString()}</td>
              <td>{DISEASE_LABELS[record.disease] || record.disease}</td>
              <td>
                {record.ml_risk_level || record.risk_level} [
                {Math.round((record.ml_risk_score ?? record.risk_score) * 100)} / 100]
              </td>
              <td>
                {record.dl_risk_level
                  ? `${record.dl_risk_level} [${Math.round(record.dl_risk_score * 100)} / 100]`
                  : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
