import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { bookApi } from "../utils/api";
import { Book } from "../types/Book";

export const BookList: React.FC = () => {
  const [books, setBooks] = useState<Book[]>([]);

  useEffect(() => {
    const fetchBooks = async () => {
      try {
        const fetchedBooks = await bookApi.getAllBooks();
        setBooks(fetchedBooks);
      } catch (error) {
        console.error("Failed to fetch books:", error);
      }
    };

    fetchBooks();
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {books.map((book) => (
        <Card key={book.id} className="p-4">
          <CardHeader>
            <CardTitle>{book.title}</CardTitle>
          </CardHeader>
          <CardContent className="flex">
            {book.book_cover_url && (
              <img
                src={book.book_cover_url}
                alt={book.title}
                className="w-32 h-48 object-cover mr-4"
              />
            )}
            <div>
              <p>Author: {book.author}</p>
              <p>Pages: {book.pages}</p>
              <p>Rating: {"⭐".repeat(book.rating)}</p>
              <p>
                Read: {book.start_date} - {book.finish_date}
              </p>
              <p>Notes: {book.notes}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};
