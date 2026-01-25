import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import MinimalPage from './components/MinimalPage'
import './App.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MinimalPage />} />
      </Routes>
    </Router>
  )
}

export default App