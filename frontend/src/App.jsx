import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Home        from "./pages/Home.jsx";
import AnimeDetail from "./pages/AnimeDetail.jsx";
import Profile     from "./pages/Profile.jsx";
import Chatbot     from "./components/Chatbot.jsx";
import AuthPanel   from "./components/AuthPanel.jsx";
import logoUrl     from "../logo/YuGen-Logo.png";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="top-bar">
          <Link to="/" className="brand">
            <img src={logoUrl} alt="YuGen" className="brand-logo" />
            <div>
              <div className="brand-title">YuGen</div>
              <div className="brand-subtitle">anime discovery hub</div>
            </div>
          </Link>
          <nav className="nav">
            <Link to="/" className="nav-link">Home</Link>
            <AuthPanel />
          </nav>
        </header>

        <Routes>
          <Route path="/"            element={<Home />} />
          <Route path="/anime/:id"   element={<AnimeDetail />} />
          <Route path="/profile"     element={<Profile />} />
        </Routes>

        <Chatbot />
      </div>
    </BrowserRouter>
  );
}
