import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="page">
      <h1>Choose a disease</h1>
      <div className="cards">
        <Link className="card" to="/predict/cvd">
          <h2>Cardiovascular</h2>
          <p>Estimate cardiovascular disease risk.</p>
        </Link>
        <Link className="card" to="/predict/diabetes">
          <h2>Type 2 Diabetes</h2>
          <p>Estimate type 2 diabetes risk.</p>
        </Link>
        <Link className="card" to="/predict/alzheimers">
          <h2>Alzheimer's disease</h2>
          <p>Estimate Alzheimer's disease risk.</p>
        </Link>
        <Link className="card" to="/predict/liver">
          <h2>Liver disease</h2>
          <p>Estimate liver disease risk.</p>
        </Link>
      </div>
    </div>
  )
}
