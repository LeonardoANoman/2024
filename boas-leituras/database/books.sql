CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    pages INTEGER,
    start_date DATE,
    finish_date DATE,
    book_cover_url TEXT,
    rating INTEGER CHECK(rating >= 0 AND rating <= 5),
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);