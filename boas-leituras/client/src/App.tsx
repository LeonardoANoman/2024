import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import { HomePage } from "./pages/HomePage";
import { AddBookPage } from "./pages/AddBookPage";
import { StatsPage } from "./pages/StatsPage";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
} from "./components/ui/navigation-menu";

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow-md">
          <div className="container mx-auto px-4 py-3">
            <NavigationMenu>
              <NavigationMenuList>
                <NavigationMenuItem>
                  <Link
                    to="/"
                    className="text-gray-800 hover:text-blue-600 transition-colors px-4 py-2"
                  >
                    My Books
                  </Link>
                </NavigationMenuItem>
                <NavigationMenuItem>
                  <Link
                    to="/add"
                    className="text-gray-800 hover:text-blue-600 transition-colors px-4 py-2"
                  >
                    Add Book
                  </Link>
                </NavigationMenuItem>
                <NavigationMenuItem>
                  <Link
                    to="/stats"
                    className="text-gray-800 hover:text-blue-600 transition-colors px-4 py-2"
                  >
                    Reading Stats
                  </Link>
                </NavigationMenuItem>
              </NavigationMenuList>
            </NavigationMenu>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/add" element={<AddBookPage />} />
          <Route path="/stats" element={<StatsPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
