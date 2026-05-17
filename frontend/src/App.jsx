import { Routes, Route, Navigate } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import Login from "./pages/Login.jsx";
import Register from "./pages/Register.jsx";
import AllAnime from "./pages/AllAnime.jsx";
import AnimeDetail from "./pages/AnimeDetail.jsx";
import ForYou from "./pages/ForYou.jsx";
import { isLoggedIn } from "./auth.js";

function Protected({ children }) {
  return isLoggedIn() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <>
      <Nav />
      <main className="container py-4">
        <Routes>
          <Route path="/" element={<Navigate to="/anime" replace />} />
          <Route path="/anime" element={<AllAnime />} />
          <Route path="/anime/:id" element={<AnimeDetail />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/for-you"
            element={
              <Protected>
                <ForYou />
              </Protected>
            }
          />
          <Route path="*" element={<p>Not found.</p>} />
        </Routes>
      </main>
    </>
  );
}
