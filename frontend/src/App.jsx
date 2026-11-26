import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { getToken } from './api'
import NavBar from './components/NavBar'
import Assessment from './pages/Assessment'
import History from './pages/History'
import Home from './pages/Home'
import Login from './pages/Login'
import Predict from './pages/Predict'
import Register from './pages/Register'

function Protected({ children }) {
  if (!getToken()) return <Navigate to="/login" replace />
  return (
    <>
      <NavBar />
      <div className="banner">Academic demonstration. Not a medical diagnosis.</div>
      {children}
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/"
          element={
            <Protected>
              <Home />
            </Protected>
          }
        />
        <Route
          path="/predict/:disease"
          element={
            <Protected>
              <Predict />
            </Protected>
          }
        />
        <Route
          path="/assessment"
          element={
            <Protected>
              <Assessment />
            </Protected>
          }
        />
        <Route
          path="/history"
          element={
            <Protected>
              <History />
            </Protected>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
