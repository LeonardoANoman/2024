import { Request, Response } from "express";
import { db } from "../index.js";
import axios from "axios";

export const addBook = async (req: Request, res: Response) => {
  try {
    const {
      title,
      author,
      pages,
      start_date,
      finish_date,
      book_cover_url,
      rating,
      notes,
    } = req.body;

    const result = await db.run(
      "INSERT INTO books (title, author, pages, start_date, finish_date, book_cover_url, rating, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      [
        title,
        author,
        pages,
        start_date,
        finish_date,
        book_cover_url,
        rating,
        notes,
      ]
    );

    res.status(201).json({ id: result.lastID });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
};

export const getAllBooks = async (_req: Request, res: Response) => {
  try {
    const books = await db.all("SELECT * FROM books ORDER BY created_at DESC");
    res.json(books);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
};

export const getBookStats = async (_req: Request, res: Response) => {
  try {
    const totalBooks = await db.get("SELECT COUNT(*) as count FROM books");
    const totalPagesRead = await db.get(
      "SELECT SUM(pages) as total FROM books"
    );
    const averageRating = await db.get("SELECT AVG(rating) as avg FROM books");

    const booksByMonth = await db.all(`
          WITH RECURSIVE 
          months(date) AS (
              SELECT date('now', 'start of month', '-11 months')
              UNION ALL
              SELECT date(date, '+1 month')
              FROM months
              WHERE date < date('now', 'start of month')
          )
          SELECT 
              strftime('%Y-%m', months.date) as month,
              strftime('%m/%Y', months.date) as month_label,
              COUNT(books.id) as book_count,
              COALESCE(SUM(books.pages), 0) as pages_read
          FROM months
          LEFT JOIN books ON strftime('%Y-%m', books.finish_date) = strftime('%Y-%m', months.date)
          GROUP BY strftime('%Y-%m', months.date)
          ORDER BY months.date ASC
          LIMIT 12
      `);

    res.json({
      totalBooks: totalBooks.count,
      totalPagesRead: totalPagesRead.total,
      averageRating: averageRating.avg,
      booksByMonth,
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
};

export const searchBookImage = async (req: Request, res: Response) => {
  const { title, author } = req.query;

  try {
    const response = await axios.get(
      "https://www.googleapis.com/books/v1/volumes",
      {
        params: {
          q: `${title} ${author}`,
          key: process.env.GOOGLE_BOOKS_API_KEY,
        },
      }
    );

    const imageUrl =
      response.data.items?.[0]?.volumeInfo?.imageLinks?.thumbnail || "";
    const secureImageUrl = imageUrl.replace("http://", "https://");

    res.json({ imageUrl: secureImageUrl });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
};
