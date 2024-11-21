import React from "react";
import { BookList } from "../components/BookList";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";

export const HomePage: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Boas Leituras</CardTitle>
        </CardHeader>
        <CardContent>
          <BookList />
        </CardContent>
      </Card>
    </div>
  );
};
