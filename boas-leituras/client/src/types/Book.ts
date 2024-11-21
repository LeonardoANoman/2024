export interface Book {
  id?: number;
  title: string;
  author: string;
  pages: number;
  start_date: string;
  finish_date: string;
  book_cover_url: string;
  rating: number;
  notes: string;
}

export interface BookStats {
  totalBooks: number;
  totalPagesRead: number;
  averageRating: number;
  booksByMonth: {
    month: string;
    month_label: string;
    book_count: number;
    pages_read: number;
  }[];
}
