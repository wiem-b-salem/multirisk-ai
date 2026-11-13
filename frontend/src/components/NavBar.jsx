import { Link, useNavigate } from 'react-router-dom'
import { clearToken } from '../api'

export default function NavBar() {
  const navigate = useNavigate()

  function logout() {
    clearToken()
    navigate('/login')
  }

  return (
    <nav className="nav">
      <Link to="/">Home</Link>
      <Link to="/history">History</Link>
      <button type="button" onClick={logout}>
        Logout
      </button>
    </nav>
  )
}
