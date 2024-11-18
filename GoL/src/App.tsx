import React, { useState, useEffect } from "react";

const GRID_SIZE = 33;
const CELL_SIZE = 20;
const TICK_INTERVAL = 200;

type Grid = boolean[][];

const generateEmptyGrid = (): Grid =>
  Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(false));

const App: React.FC = () => {
  const [grid, setGrid] = useState<Grid>(generateEmptyGrid());
  const [isRunning, setIsRunning] = useState(false);

  const toggleCell = (row: number, col: number) => {
    const newGrid = grid.map((row) => [...row]);
    newGrid[row][col] = !newGrid[row][col];
    setGrid(newGrid);
  };

  const getNextGeneration = (grid: Grid): Grid => {
    const nextGrid = generateEmptyGrid();
    const directions = [
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];

    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const liveNeighbors = directions.reduce((acc, [dx, dy]) => {
          const newRow = row + dx;
          const newCol = col + dy;
          if (
            newRow >= 0 &&
            newRow < GRID_SIZE &&
            newCol >= 0 &&
            newCol < GRID_SIZE &&
            grid[newRow][newCol]
          ) {
            return acc + 1;
          }
          return acc;
        }, 0);

        if (grid[row][col]) {
          nextGrid[row][col] = liveNeighbors === 2 || liveNeighbors === 3;
        } else {
          nextGrid[row][col] = liveNeighbors === 3;
        }
      }
    }

    return nextGrid;
  };

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setGrid((prevGrid) => getNextGeneration(prevGrid));
    }, TICK_INTERVAL);
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Game of Life</h1>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${GRID_SIZE}, ${CELL_SIZE}px)`,
          margin: "20px auto",
          border: "2px solid #444",
          boxShadow: "4px 8px rgba(0, 0, 0, 0.1)",
        }}
      >
        {grid.map((row, rowIndex) =>
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              onClick={() => toggleCell(rowIndex, colIndex)}
              style={{
                ...styles.cell,
                backgroundColor: cell ? "#4CAF50" : "#F1F1F1",
                transition: "background-color 0.2s",
              }}
            />
          ))
        )}
      </div>
      <div style={styles.buttonContainer}>
        <button onClick={() => setIsRunning(!isRunning)} style={styles.button}>
          {isRunning ? "Stop" : "Start"}
        </button>
        <button
          onClick={() => setGrid(generateEmptyGrid())}
          style={{ ...styles.button, backgroundColor: "#e57373" }}
        >
          Clear
        </button>
        <button
          onClick={() => setGrid((prevGrid) => getNextGeneration(prevGrid))}
          style={{ ...styles.button, backgroundColor: "#64b5f6" }}
        >
          Next
        </button>
      </div>
    </div>
  );
};

const styles = {
  container: {
    textAlign: "center" as const,
    padding: "20px 600px",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  title: {
    fontSize: "2.5rem",
    color: "#333",
    marginBottom: "20px",
  },
  cell: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    border: "1px solid #ddd",
    cursor: "pointer",
  },
  buttonContainer: {
    marginTop: "20px",
    display: "flex",
    justifyContent: "center",
    gap: "10px",
  },
  button: {
    padding: "10px 20px",
    fontSize: "1rem",
    cursor: "pointer",
    border: "none",
    borderRadius: "5px",
    backgroundColor: "#81c784",
    color: "#fff",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    transition: "background-color 0.3s, transform 0.1s",
  },
};

export default App;
