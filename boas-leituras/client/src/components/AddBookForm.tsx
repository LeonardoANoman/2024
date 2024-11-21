import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Book } from "../types/Book";
import { bookApi } from "../utils/api";

export const AddBookForm: React.FC = () => {
  const [bookData, setBookData] = useState<Book>({
    title: "",
    author: "",
    pages: 0,
    start_date: "",
    finish_date: "",
    book_cover_url: "",
    rating: 0,
    notes: "",
  });

  const handleImageSearch = async () => {
    try {
      const imageUrl = await bookApi.searchBookImage(
        bookData.title,
        bookData.author
      );
      setBookData((prev) => ({ ...prev, book_cover_url: imageUrl }));
    } catch (error) {
      console.error("Image search failed", error);
    }
  };

  const handleSubmit = async () => {
    try {
      await bookApi.addBook(bookData);
      // Reset form or show success message
      setBookData({
        title: "",
        author: "",
        pages: 0,
        start_date: "",
        finish_date: "",
        book_cover_url: "",
        rating: 0,
        notes: "",
      });
    } catch (error) {
      console.error("Book submission failed", error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Add New Book</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Title"
          value={bookData.title}
          onChange={(e) =>
            setBookData((prev) => ({ ...prev, title: e.target.value }))
          }
        />
        <Input
          placeholder="Author"
          value={bookData.author}
          onChange={(e) =>
            setBookData((prev) => ({ ...prev, author: e.target.value }))
          }
        />
        <Input
          type="number"
          placeholder="Number of Pages"
          value={bookData.pages}
          onChange={(e) =>
            setBookData((prev) => ({
              ...prev,
              pages: parseInt(e.target.value),
            }))
          }
        />
        <Input
          type="date"
          placeholder="Start Date"
          value={bookData.start_date}
          onChange={(e) =>
            setBookData((prev) => ({ ...prev, start_date: e.target.value }))
          }
        />
        <Input
          type="date"
          placeholder="Finish Date"
          value={bookData.finish_date}
          onChange={(e) =>
            setBookData((prev) => ({ ...prev, finish_date: e.target.value }))
          }
        />
        <Button onClick={handleImageSearch}>Search Book Image</Button>

        <Select
          value={bookData.rating.toString()}
          onValueChange={(val) =>
            setBookData((prev) => ({ ...prev, rating: parseInt(val) }))
          }
        >
          <SelectTrigger>
            <SelectValue placeholder="Rating" />
          </SelectTrigger>
          <SelectContent>
            {[1, 2, 3, 4, 5].map((num) => (
              <SelectItem key={num} value={num.toString()}>
                {"⭐".repeat(num)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          placeholder="Notes"
          value={bookData.notes}
          onChange={(e) =>
            setBookData((prev) => ({ ...prev, notes: e.target.value }))
          }
        />

        <Button onClick={handleSubmit}>Add Book</Button>
      </CardContent>
    </Card>
  );
};
