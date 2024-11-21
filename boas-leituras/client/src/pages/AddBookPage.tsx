import React from "react";
import { AddBookForm } from "../components/AddBookForm";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";

export const AddBookPage: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Add New Book</CardTitle>
        </CardHeader>
        <CardContent>
          <AddBookForm />
        </CardContent>
      </Card>
    </div>
  );
};
