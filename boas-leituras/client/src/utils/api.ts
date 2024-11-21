import axios from "axios";
import { Book, BookStats } from "../types/Book";

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000/api";

export const bookApi = {
  getAllBooks: async (): Promise<Book[]> => {
    const response = await axios.get(`${API_BASE_URL}/books`);
    return response.data;
  },

  addBook: async (book: Book): Promise<Book> => {
    const response = await axios.post(`${API_BASE_URL}/books`, book);
    return response.data;
  },

  searchBookImage: async (title: string, author: string): Promise<string> => {
    const response = await axios.get(`${API_BASE_URL}/books/search-image`, {
      params: { title, author },
    });
    return response.data.imageUrl;
  },

  getBookStats: async (): Promise<BookStats> => {
    const response = await axios.get(`${API_BASE_URL}/books/stats`);
    return response.data;
  },
};
