import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Game from "./PrimeMatrixGame.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Game />
  </StrictMode>
);
