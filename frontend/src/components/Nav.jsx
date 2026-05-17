import { Link, NavLink, useNavigate } from "react-router-dom";
import { clearToken, isLoggedIn } from "../auth.js";

export default function Nav() {
  const navigate = useNavigate();
  const loggedIn = isLoggedIn();

  function logout() {
    clearToken();
    navigate("/login");
  }

  const navLinkClass = ({ isActive }) =>
    "nav-link" + (isActive ? " active" : "");

  return (
    <nav className="navbar navbar-expand-md bg-body-tertiary border-bottom">
      <div className="container">
        <Link className="navbar-brand fw-semibold" to="/">
          YuGen
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarMain"
        >
          <span className="navbar-toggler-icon" />
        </button>
        <div className="collapse navbar-collapse" id="navbarMain">
          <ul className="navbar-nav me-auto">
            <li className="nav-item">
              <NavLink className={navLinkClass} to="/anime">
                Browse
              </NavLink>
            </li>
            {loggedIn && (
              <li className="nav-item">
                <NavLink className={navLinkClass} to="/for-you">
                  For you
                </NavLink>
              </li>
            )}
          </ul>
          <ul className="navbar-nav">
            {loggedIn ? (
              <li className="nav-item">
                <button className="btn btn-outline-secondary btn-sm" onClick={logout}>
                  Log out
                </button>
              </li>
            ) : (
              <>
                <li className="nav-item">
                  <NavLink className={navLinkClass} to="/login">
                    Log in
                  </NavLink>
                </li>
                <li className="nav-item">
                  <NavLink className={navLinkClass} to="/register">
                    Register
                  </NavLink>
                </li>
              </>
            )}
          </ul>
        </div>
      </div>
    </nav>
  );
}
