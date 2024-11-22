import { useState, useEffect } from "react";
import "./styles.css";

const PrimeMatrixGame = () => {
  const [size, setSize] = useState(2);
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [guess, setGuess] = useState("");
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState("");
  const [actualPrimes, setActualPrimes] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [usedNumbers, setUsedNumbers] = useState<Set<number>>(new Set());

  const isPrime = (num: number): boolean => {
    if (num <= 1) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
      if (num % i === 0) return false;
    }
    return true;
  };

  const getNextOddNumber = (current: number): number => {
    let next = current + 2;
    while (usedNumbers.has(next)) {
      next += 2;
    }
    return next;
  };

  const generateMatrix = (size: number) => {
    const newMatrix: number[][] = [];
    let primeCount = 0;
    let currentNum = 1;

    for (let i = 0; i < size; i++) {
      const row: number[] = [];
      for (let j = 0; j < size; j++) {
        currentNum = getNextOddNumber(currentNum);
        row.push(currentNum);
        usedNumbers.add(currentNum);
        if (isPrime(currentNum)) primeCount++;
      }
      newMatrix.push(row);
    }

    setMatrix(newMatrix);
    setActualPrimes(primeCount);
    setShowAnswer(false);
    setGuess("");
    setFeedback("");
  };

  useEffect(() => {
    generateMatrix(size);
  }, [size]);

  const handleGuess = () => {
    const guessNum = parseInt(guess);
    if (guessNum === actualPrimes) {
      setScore(score + 1);
      setFeedback("Correct! Moving to next level...");
      setTimeout(() => {
        setSize(size + 1);
      }, 1500);
    } else {
      setFeedback("Wrong answer! Try again");
      setShowAnswer(true);
    }
  };

  return (
    <div className="game-card">
      <div className="game-header">
        <h2>Prime Matrix Challenge</h2>
        <div className="score">
          <span className="trophy">🏆</span>
          <span>Score: {score}</span>
        </div>
      </div>

      <div className="game-content">
        <div className="matrix">
          {matrix.map((row, i) => (
            <div key={i} className="matrix-row">
              {row.map((num, j) => (
                <div
                  key={`${i}-${j}`}
                  className={`matrix-cell ${
                    showAnswer && isPrime(num) ? "prime" : ""
                  }`}
                >
                  {num}
                </div>
              ))}
            </div>
          ))}
        </div>

        <div className="controls">
          <input
            type="number"
            value={guess}
            onChange={(e) => setGuess(e.target.value)}
            placeholder="Number of primes?"
            className="guess-input"
          />
          <button
            onClick={handleGuess}
            disabled={!guess}
            className="submit-button"
          >
            Submit →
          </button>
        </div>

        {feedback && (
          <div
            className={`feedback ${
              feedback.includes("Correct") ? "correct" : "wrong"
            }`}
          >
            {feedback}
            {showAnswer && ` (Answer: ${actualPrimes})`}
          </div>
        )}
      </div>
    </div>
  );
};

export default PrimeMatrixGame;
